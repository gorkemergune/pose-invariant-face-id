"""Descriptive dataset figures for the README (no face images, license-safe).

Reads results/pose_labels.csv and produces two aggregate charts that explain the
FEI pose structure without redistributing any face image:

  results/dataset_pose_structure.png  mean yaw (+/- std) per FEI image index,
                                       showing the ~180-degree rotation sweep
  results/pose_bin_distribution.png   count of frontal / half-profile / profile

Usage:
    python src/dataset_overview.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSE_ORDER = ["frontal", "half-profile", "profile"]
POSE_COLORS = {"frontal": "#4c72b0", "half-profile": "#dd8452", "profile": "#c44e52"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose-labels", type=Path,
                   default=PROJECT_ROOT / "results" / "pose_labels.csv")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    yaw_by_idx: dict[int, list[float]] = defaultdict(list)
    bin_counts: dict[str, int] = defaultdict(int)
    with args.pose_labels.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok":
                continue
            yaw_by_idx[int(r["image_num"])].append(float(r["yaw"]))
            bin_counts[r["pose_bin"]] += 1

    # --- Figure 1: yaw sweep across the 14 FEI image indices ---
    idxs = sorted(yaw_by_idx)
    means = [np.mean(yaw_by_idx[i]) for i in idxs]
    stds = [np.std(yaw_by_idx[i]) for i in idxs]
    plt.figure(figsize=(9, 5))
    plt.axhspan(-20, 20, color="#4c72b0", alpha=0.10, label="frontal band (|yaw|<20°)")
    plt.errorbar(idxs, means, yerr=stds, fmt="o-", color="#333", ecolor="#999",
                 capsize=3, lw=1.8, label="mean yaw ± std")
    for i, m in zip(idxs, means):
        plt.annotate(f"{m:.0f}°", (i, m), textcoords="offset points",
                     xytext=(0, 8), fontsize=7, ha="center")
    plt.xticks(idxs)
    plt.xlabel("FEI image index (per subject)")
    plt.ylabel("estimated yaw (degrees)")
    plt.title("FEI pose structure: yaw sweep across the 14 images per subject\n"
              "(images 1–10 rotate left→right; 11–14 are frontal expression/illumination shots)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    f1 = args.out_dir / "dataset_pose_structure.png"
    plt.savefig(f1, dpi=150)
    plt.close()

    # --- Figure 2: pose-bin distribution ---
    counts = [bin_counts[b] for b in POSE_ORDER]
    total = sum(counts)
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(POSE_ORDER, counts, color=[POSE_COLORS[b] for b in POSE_ORDER])
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, c + 8,
                 f"{c}\n({100*c/total:.1f}%)", ha="center", fontsize=9)
    plt.ylabel("number of images")
    plt.title(f"Pose-bin distribution ({total} detected images)")
    plt.ylim(0, max(counts) * 1.18)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    f2 = args.out_dir / "pose_bin_distribution.png"
    plt.savefig(f2, dpi=150)
    plt.close()

    print(f"Saved: {f1}\n       {f2}")


if __name__ == "__main__":
    main()
