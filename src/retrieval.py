"""FAISS retrieval evaluation on the test split.

Builds a FAISS inner-product index over the L2-normalized ArcFace embeddings of
the test-split images. Each test image is used in turn as a query against a
gallery of all other test images (the query itself is excluded). A retrieval is
correct if a *same-identity* image appears in the top-k neighbours -- i.e. the
gallery holds the same person at different poses, and we ask whether the model
can find them.

Reports top-1 and top-5 accuracy overall and broken down by the QUERY image's
pose bin (frontal / half-profile / profile), which shows how retrieval degrades
as the query moves off-frontal.

Output: results/retrieval_metrics.csv

Usage:
    python src/retrieval.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import faiss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSE_ORDER = ["frontal", "half-profile", "profile"]
TOPK = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", type=Path,
                   default=PROJECT_ROOT / "data" / "processed" / "embeddings.npz")
    p.add_argument("--split", type=Path,
                   default=PROJECT_ROOT / "results" / "split_assignment.csv")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "results" / "retrieval_metrics.csv")
    return p.parse_args()


def load_test(emb_path: Path, split_path: Path):
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
    return np.ascontiguousarray(emb[mask]), subj[mask], bins[mask]


def crosspose_eval(emb, subj, bins, out_csv: Path, out_png: Path) -> list[dict]:
    """Frontal-only gallery; half-profile and profile images as queries.

    Measures true pose-invariant retrieval: can an off-frontal query find the
    same identity among frontal-only gallery images?
    """
    fmask = bins == "frontal"
    gallery_emb = np.ascontiguousarray(emb[fmask])
    gallery_subj = subj[fmask].astype(int)
    gallery_ids = set(gallery_subj.tolist())

    index = faiss.IndexFlatIP(gallery_emb.shape[1])
    index.add(gallery_emb)

    print("\n=== Cross-pose retrieval (gallery = frontal only) ===")
    print(f"gallery: {gallery_emb.shape[0]} frontal images, "
          f"{len(gallery_ids)} identities")
    missing = set(subj.astype(int).tolist()) - gallery_ids
    if missing:
        print(f"  [warn] identities with no frontal gallery image (queries excluded): "
              f"{sorted(missing)}")

    rows = []
    agg = {}
    for pose in ("half-profile", "profile"):
        qmask = bins == pose
        q_emb = np.ascontiguousarray(emb[qmask])
        q_subj = subj[qmask].astype(int)
        keep = np.array([s in gallery_ids for s in q_subj])
        q_emb, q_subj = q_emb[keep], q_subj[keep]
        if len(q_subj) == 0:
            continue
        _, idx = index.search(q_emb, TOPK)
        top1 = top5 = 0
        for i in range(len(q_subj)):
            neigh = [int(gallery_subj[j]) for j in idx[i]]
            top1 += int(neigh[0] == q_subj[i])
            top5 += int(q_subj[i] in neigh)
        n = len(q_subj)
        agg[pose] = (n, top1 / n, top5 / n)
        rows.append({"scope": pose, "n_queries": n,
                     "top1_acc": round(top1 / n, 4), "top5_acc": round(top5 / n, 4)})

    # combined non-frontal
    if agg:
        tot_n = sum(v[0] for v in agg.values())
        tot1 = sum(v[0] * v[1] for v in agg.values())
        tot5 = sum(v[0] * v[2] for v in agg.values())
        rows.append({"scope": "all-nonfrontal", "n_queries": tot_n,
                     "top1_acc": round(tot1 / tot_n, 4), "top5_acc": round(tot5 / tot_n, 4)})

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "n_queries", "top1_acc", "top5_acc"])
        w.writeheader()
        w.writerows(rows)

    # bar chart: top1/top5 by query pose bin
    labels = [r["scope"] for r in rows]
    x = np.arange(len(labels))
    t1 = [r["top1_acc"] for r in rows]
    t5 = [r["top5_acc"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.bar(x - 0.2, t1, width=0.4, label="top-1", color="#4c72b0")
    plt.bar(x + 0.2, t5, width=0.4, label="top-5", color="#dd8452")
    for xi, (a, b) in enumerate(zip(t1, t5)):
        plt.text(xi - 0.2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        plt.text(xi + 0.2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
    plt.xticks(x, labels)
    plt.ylim(0, 1.08)
    plt.ylabel("retrieval accuracy")
    plt.title("Cross-pose retrieval (frontal-only gallery) by query pose")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"{'scope':<16}{'n_queries':>10}{'top1':>9}{'top5':>9}")
    for r in rows:
        print(f"{r['scope']:<16}{r['n_queries']:>10}{r['top1_acc']:>9.3f}{r['top5_acc']:>9.3f}")
    print(f"Saved -> {out_csv}\n         {out_png}")
    return rows


def main() -> None:
    args = parse_args()
    emb, subj, bins = load_test(args.embeddings, args.split)
    n, dim = emb.shape
    print(f"Test gallery: {n} images, {len(set(int(s) for s in subj))} identities")

    index = faiss.IndexFlatIP(dim)  # cosine = inner product on unit vectors
    index.add(emb)

    # Search top (k+1): the first hit is the query itself (self-match), dropped.
    k = TOPK + 1
    _, idx = index.search(emb, k)

    # per-query correctness, aggregated overall and by query pose bin
    agg = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0})

    # Guard: only queries whose identity has >=2 test images can be retrieved.
    counts = defaultdict(int)
    for s in subj:
        counts[int(s)] += 1

    for i in range(n):
        qid = int(subj[i])
        if counts[qid] < 2:
            continue  # no same-identity gallery image exists
        neighbours = [j for j in idx[i] if j != i][:TOPK]
        neigh_ids = [int(subj[j]) for j in neighbours]
        top1 = neigh_ids[0] == qid
        top5 = qid in neigh_ids
        for scope in ("overall", bins[i]):
            agg[scope]["n"] += 1
            agg[scope]["top1"] += int(top1)
            agg[scope]["top5"] += int(top5)

    rows = []
    for scope in ["overall"] + POSE_ORDER:
        a = agg[scope]
        if a["n"] == 0:
            continue
        rows.append({
            "scope": scope,
            "n_queries": a["n"],
            "top1_acc": round(a["top1"] / a["n"], 4),
            "top5_acc": round(a["top5"] / a["n"], 4),
        })

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "n_queries", "top1_acc", "top5_acc"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== Retrieval accuracy (query vs gallery of other test images) ===")
    print(f"{'scope':<16}{'n_queries':>10}{'top1':>9}{'top5':>9}")
    for r in rows:
        print(f"{r['scope']:<16}{r['n_queries']:>10}{r['top1_acc']:>9.3f}{r['top5_acc']:>9.3f}")
    print(f"\nSaved -> {args.out}")

    crosspose_eval(
        emb, subj, bins,
        out_csv=args.out.parent / "retrieval_crosspose_metrics.csv",
        out_png=args.out.parent / "retrieval_crosspose.png",
    )


if __name__ == "__main__":
    main()
