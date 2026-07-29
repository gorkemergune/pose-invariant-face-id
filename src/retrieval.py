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


if __name__ == "__main__":
    main()
