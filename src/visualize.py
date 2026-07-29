"""2D visualization of ArcFace embeddings (UMAP and t-SNE).

Reduces the TEST-split embeddings to 2D with both UMAP and t-SNE. For each
method a figure is written with two panels sharing the same layout:
  (left)  colored by subject_id  -- only the 20 test identities, to avoid
          clutter and to show identity clustering;
  (right) colored by pose_bin    -- frontal / half-profile / profile.

Using the same points colored two ways makes the comparison direct: if identity
clusters are cleaner than pose clusters, the embedding encodes identity more
strongly than pose (the desirable property for a recognition model).

A silhouette score (cosine, on the raw 512-d vectors) is also printed for
identity vs pose labels as an objective backup to the visual read.

Outputs:
  results/embedding_umap.png
  results/embedding_tsne.png

Usage:
    python src/visualize.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import umap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSE_ORDER = ["frontal", "half-profile", "profile"]
POSE_COLORS = {"frontal": "#4c72b0", "half-profile": "#dd8452", "profile": "#c44e52"}
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", type=Path,
                   default=PROJECT_ROOT / "data" / "processed" / "embeddings.npz")
    p.add_argument("--split", type=Path,
                   default=PROJECT_ROOT / "results" / "split_assignment.csv")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    return p.parse_args()


def load_test_embeddings(emb_path: Path, split_path: Path):
    split = {}
    with split_path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            split[int(r["subject_id"])] = r["split"]

    d = np.load(emb_path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    subj = d["subject_ids"]
    bins = d["pose_bins"]

    mask = np.array([split.get(int(s)) == "test" for s in subj])
    return emb[mask], subj[mask], bins[mask]


def scatter_panels(xy: np.ndarray, subj: np.ndarray, bins: np.ndarray,
                   method: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # (1) color by identity (20 test subjects)
    ids = sorted(set(int(s) for s in subj))
    cmap = plt.get_cmap("tab20", len(ids))
    for i, sid in enumerate(ids):
        m = subj == sid
        ax1.scatter(xy[m, 0], xy[m, 1], s=28, color=cmap(i), label=str(sid),
                    edgecolors="none", alpha=0.85)
    ax1.set_title(f"{method}: colored by subject_id (20 test identities)")
    ax1.legend(title="subject", fontsize=6, ncol=2, loc="best", framealpha=0.6)

    # (2) color by pose bin
    for b in POSE_ORDER:
        m = bins == b
        ax2.scatter(xy[m, 0], xy[m, 1], s=28, color=POSE_COLORS[b], label=b,
                    edgecolors="none", alpha=0.8)
    ax2.set_title(f"{method}: colored by pose_bin")
    ax2.legend(title="pose", fontsize=9, loc="best")

    for ax in (ax1, ax2):
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"ArcFace embeddings, {method} 2D projection (test split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    emb, subj, bins = load_test_embeddings(args.embeddings, args.split)
    print(f"Test-split embeddings: {emb.shape[0]} points, "
          f"{len(set(int(s) for s in subj))} identities")

    # Objective separation measure on the raw 512-d vectors (cosine).
    sil_id = silhouette_score(emb, subj, metric="cosine")
    sil_pose = silhouette_score(emb, bins, metric="cosine")
    print("\n=== Silhouette score (cosine, 512-d) — higher = better separated ===")
    print(f"  by identity : {sil_id:.3f}")
    print(f"  by pose_bin : {sil_pose:.3f}")

    print("\nComputing UMAP ...")
    umap_xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=SEED).fit_transform(emb)
    scatter_panels(umap_xy, subj, bins, "UMAP", args.out_dir / "embedding_umap.png")

    print("Computing t-SNE ...")
    tsne_xy = TSNE(n_components=2, perplexity=30, metric="cosine",
                   init="random", random_state=SEED).fit_transform(emb)
    scatter_panels(tsne_xy, subj, bins, "t-SNE", args.out_dir / "embedding_tsne.png")

    print(f"\nSaved: {args.out_dir / 'embedding_umap.png'}")
    print(f"       {args.out_dir / 'embedding_tsne.png'}")


if __name__ == "__main__":
    main()
