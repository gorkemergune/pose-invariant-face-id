"""Identity-disjoint train/val/test split and verification pair generation.

The 200 FEI subjects are partitioned (fixed seed) into ~160 train / 20 val /
20 test *identities*. The split is identity-disjoint: a subject appears in
exactly one split, so val/test identities are never seen during training or
fine-tuning.

For each split we build verification pairs:
  * Positives  -> two images of the SAME subject (all C(n,2) within-subject
    pairs). Their pose-bin combination follows the data naturally.
  * Negatives  -> two images of DIFFERENT subjects, sampled so that the six
    unordered pose-bin combinations
        frontal-frontal, frontal-half, frontal-profile,
        half-half,       half-profile, profile-profile
    are represented as evenly as possible. The number of negatives per split
    matches the number of positives (balanced labels).

Inputs:
  results/preprocess_log.csv  (aligned_path per image, status)
  results/pose_labels.csv     (pose_bin per image, status)

Outputs:
  results/split_assignment.csv                 (subject_id, split)
  results/pairs_{train,val,test}.csv           (img1_path, img2_path, label,
                                                pose_bin1, pose_bin2)

Usage:
    python src/split.py
    python src/split.py --seed 42 --n-val 20 --n-test 20
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BINS = ("frontal", "half-profile", "profile")
# The six unordered pose-bin combinations to balance across negatives.
COMBOS = [
    ("frontal", "frontal"),
    ("frontal", "half-profile"),
    ("frontal", "profile"),
    ("half-profile", "half-profile"),
    ("half-profile", "profile"),
    ("profile", "profile"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preprocess-log", type=Path,
                   default=PROJECT_ROOT / "results" / "preprocess_log.csv")
    p.add_argument("--pose-labels", type=Path,
                   default=PROJECT_ROOT / "results" / "pose_labels.csv")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--n-test", type=int, default=20)
    return p.parse_args()


def load_images(preprocess_log: Path, pose_labels: Path) -> dict[int, list[dict]]:
    """Return {subject_id: [ {filename, path, pose_bin}, ... ]} for usable images.

    Usable = detected/aligned in preprocessing AND pose-estimated (status ok in
    both files). Joined on filename.
    """
    aligned = {}
    with preprocess_log.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] == "ok":
                aligned[r["filename"]] = r["aligned_path"]

    by_subject: dict[int, list[dict]] = defaultdict(list)
    with pose_labels.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            fn = r["filename"]
            if fn not in aligned:
                continue
            by_subject[int(r["subject_id"])].append(
                {"filename": fn, "path": aligned[fn], "pose_bin": r["pose_bin"]}
            )
    return by_subject


def assign_splits(subjects: list[int], n_val: int, n_test: int,
                  rng: random.Random) -> dict[int, str]:
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    test = set(shuffled[:n_test])
    val = set(shuffled[n_test:n_test + n_val])
    assignment = {}
    for s in subjects:
        assignment[s] = "test" if s in test else "val" if s in val else "train"
    return assignment


def positive_pairs(imgs_by_subject: dict[int, list[dict]]) -> list[dict]:
    """All within-subject image pairs."""
    pairs = []
    for imgs in imgs_by_subject.values():
        for a, b in itertools.combinations(imgs, 2):
            pairs.append({
                "img1_path": a["path"], "img2_path": b["path"],
                "label": 1, "pose_bin1": a["pose_bin"], "pose_bin2": b["pose_bin"],
            })
    return pairs


def negative_pairs(imgs_by_subject: dict[int, list[dict]], n_target: int,
                   rng: random.Random) -> list[dict]:
    """Cross-subject pairs, balanced across the six pose-bin combinations."""
    # Pool of (subject, img) grouped by pose bin.
    by_bin: dict[str, list[tuple[int, dict]]] = {b: [] for b in BINS}
    for subj, imgs in imgs_by_subject.items():
        for im in imgs:
            by_bin[im["pose_bin"]].append((subj, im))

    per_combo = n_target // len(COMBOS)
    remainder = n_target - per_combo * len(COMBOS)
    targets = {c: per_combo + (1 if i < remainder else 0)
               for i, c in enumerate(COMBOS)}

    seen: set[frozenset[str]] = set()
    pairs: list[dict] = []

    for combo in COMBOS:
        b1, b2 = combo
        pool1, pool2 = by_bin[b1], by_bin[b2]
        want = targets[combo]
        got = 0
        # Cap attempts to avoid an infinite loop if a combo is under-populated.
        max_attempts = max(2000, want * 60)
        attempts = 0
        while got < want and attempts < max_attempts:
            attempts += 1
            s1, im1 = rng.choice(pool1)
            s2, im2 = rng.choice(pool2)
            if s1 == s2:
                continue
            key = frozenset((im1["path"], im2["path"]))
            if len(key) == 1 or key in seen:
                continue
            seen.add(key)
            pairs.append({
                "img1_path": im1["path"], "img2_path": im2["path"],
                "label": 0, "pose_bin1": im1["pose_bin"], "pose_bin2": im2["pose_bin"],
            })
            got += 1
        if got < want:
            print(f"    [warn] combo {b1}/{b2}: wanted {want}, produced {got} "
                  f"(pool sizes {len(pool1)}/{len(pool2)})")
    return pairs


def write_pairs(path: Path, pairs: list[dict]) -> None:
    fields = ["img1_path", "img2_path", "label", "pose_bin1", "pose_bin2"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(pairs)


def combo_distribution(pairs: list[dict]) -> dict[tuple[str, str], int]:
    dist: dict[tuple[str, str], int] = defaultdict(int)
    for p in pairs:
        dist[tuple(sorted((p["pose_bin1"], p["pose_bin2"])))] += 1
    return dist


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    by_subject = load_images(args.preprocess_log, args.pose_labels)
    subjects = sorted(by_subject)
    n_train = len(subjects) - args.n_val - args.n_test
    print(f"Usable subjects: {len(subjects)} "
          f"(train {n_train} / val {args.n_val} / test {args.n_test}), seed={args.seed}")

    assignment = assign_splits(subjects, args.n_val, args.n_test, rng)
    with (args.out_dir / "split_assignment.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subject_id", "split"])
        for s in subjects:
            w.writerow([s, assignment[s]])

    split_names = ("train", "val", "test")
    print("\n=== Pair generation ===")
    for split in split_names:
        imgs = {s: by_subject[s] for s in subjects if assignment[s] == split}
        pos = positive_pairs(imgs)
        neg = negative_pairs(imgs, n_target=len(pos), rng=rng)
        all_pairs = pos + neg
        rng.shuffle(all_pairs)
        write_pairs(args.out_dir / f"pairs_{split}.csv", all_pairs)

        n_img = sum(len(v) for v in imgs.values())
        print(f"\n[{split}]  subjects={len(imgs)}  images={n_img}")
        print(f"  positives={len(pos)}  negatives={len(neg)}  total={len(all_pairs)}")
        print("  negative pose-bin combo distribution:")
        ndist = combo_distribution(neg)
        for c in COMBOS:
            print(f"    {c[0]:>12} / {c[1]:<12}: {ndist.get(c, 0)}")
        print("  positive pose-bin combo distribution:")
        pdist = combo_distribution(pos)
        for c in COMBOS:
            print(f"    {c[0]:>12} / {c[1]:<12}: {pdist.get(c, 0)}")

    print(f"\nSplit assignment -> {args.out_dir / 'split_assignment.csv'}")
    print(f"Pair lists       -> {args.out_dir}/pairs_[train|val|test].csv")


if __name__ == "__main__":
    main()
