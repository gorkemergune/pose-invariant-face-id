"""Pose-aware face-verification evaluation.

Protocol (strict val -> test, no test tuning):
  1. Score every pair by cosine similarity of the two ArcFace embeddings.
     Pairs whose either image has no embedding are skipped (guard) and counted.
  2. On the VAL split, compute the ROC/AUC and pick two fixed score thresholds:
     one giving FAR = 1e-3 and one giving FAR = 1e-2.
  3. Freeze those thresholds and apply them to the TEST split. Report AUC, EER,
     TAR@FAR (with the realized test FAR) and accuracy@threshold on test. Val is
     used only for threshold selection; nothing is tuned on test.
  4. Break results down by the six unordered pose-bin combinations, using the
     same frozen thresholds, to show how accuracy varies with pose.
  5. For every scope, attach a 1000-resample bootstrap 95% CI on AUC; this is
     especially important for the sparse profile-profile bin.

Outputs:
  results/roc_curve.png            overall + per pose-bin ROC (test)
  results/verification_metrics.csv full metrics table

Usage:
    python src/evaluate.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMBOS = [
    ("frontal", "frontal"),
    ("frontal", "half-profile"),
    ("frontal", "profile"),
    ("half-profile", "half-profile"),
    ("half-profile", "profile"),
    ("profile", "profile"),
]
COMBO_LABEL = {c: f"{c[0]}/{c[1]}" for c in COMBOS}
FAR_TARGETS = [1e-3, 1e-2]
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", type=Path,
                   default=PROJECT_ROOT / "data" / "processed" / "embeddings.npz")
    p.add_argument("--pairs-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results")
    return p.parse_args()


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12  # L2-normalize
    return dict(zip(d["image_paths"].tolist(), emb))


def score_pairs(pairs_csv: Path, emb: dict[str, np.ndarray]):
    """Return (scores, labels, combos, n_skipped) with the missing-embedding guard."""
    scores, labels, combos = [], [], []
    skipped = 0
    with pairs_csv.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p1, p2 = r["img1_path"], r["img2_path"]
            if p1 not in emb or p2 not in emb:  # guard
                skipped += 1
                continue
            scores.append(float(emb[p1] @ emb[p2]))  # cosine (unit vectors)
            labels.append(int(r["label"]))
            combos.append(tuple(sorted((r["pose_bin1"], r["pose_bin2"]))))
    return np.array(scores), np.array(labels), combos, skipped


def threshold_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> float:
    """Largest-TAR operating point on VAL with FAR <= target_far."""
    fpr, tpr, thr = roc_curve(labels, scores)
    ok = np.where(fpr <= target_far)[0]
    idx = ok[-1] if len(ok) else int(np.argmin(fpr))
    return float(thr[idx])


def eer(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def at_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = scores >= thr
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = N - fp
    return {
        "tar": tp / P if P else float("nan"),
        "far": fp / N if N else float("nan"),
        "acc": (tp + tn) / (P + N) if (P + N) else float("nan"),
    }


def bootstrap_auc_ci(scores: np.ndarray, labels: np.ndarray,
                     reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED):
    """Percentile 95% CI on AUC via pair resampling with replacement."""
    if len(np.unique(labels)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        yl = labels[idx]
        if len(np.unique(yl)) < 2:
            continue
        aucs.append(roc_auc_score(yl, scores[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def scope_metrics(scores, labels, thresholds: dict) -> dict:
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    row = {"n_pairs": len(labels), "n_pos": P, "n_neg": N}
    row["pos_sim_mean"] = float(scores[labels == 1].mean()) if P else float("nan")
    row["neg_sim_mean"] = float(scores[labels == 0].mean()) if N else float("nan")
    if len(np.unique(labels)) == 2:
        row["auc"] = float(roc_auc_score(labels, scores))
        row["eer"] = eer(scores, labels)
        lo, hi = bootstrap_auc_ci(scores, labels)
        row["auc_ci95_low"], row["auc_ci95_high"] = lo, hi
    else:
        row["auc"] = row["eer"] = row["auc_ci95_low"] = row["auc_ci95_high"] = float("nan")
    for far, thr in thresholds.items():
        m = at_threshold(scores, labels, thr)
        tag = f"far{far:g}"
        row[f"tar@{tag}"] = m["tar"]
        row[f"realized_far@{tag}"] = m["far"]
        row[f"acc@{tag}"] = m["acc"]
    return row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    emb = load_embeddings(args.embeddings)

    val_s, val_y, _, val_skip = score_pairs(args.pairs_dir / "pairs_val.csv", emb)
    test_s, test_y, test_combos, test_skip = score_pairs(args.pairs_dir / "pairs_test.csv", emb)
    test_combos = np.array([COMBO_LABEL.get(c, "/".join(c)) for c in test_combos])

    print("=== Missing-embedding guard ===")
    print(f"  val pairs skipped : {val_skip}")
    print(f"  test pairs skipped: {test_skip}")

    # --- Threshold selection on VAL only ---
    thresholds = {far: threshold_at_far(val_s, val_y, far) for far in FAR_TARGETS}
    val_auc = float(roc_auc_score(val_y, val_s))
    print("\n=== Threshold selection (VAL) ===")
    print(f"  val AUC = {val_auc:.4f}")
    for far, thr in thresholds.items():
        print(f"  FAR={far:g} -> cosine threshold = {thr:.4f}")

    # --- Apply frozen thresholds to TEST ---
    rows = []
    overall = {"scope": "overall"}
    overall.update(scope_metrics(test_s, test_y, thresholds))
    rows.append(overall)

    for c in COMBOS:
        label = COMBO_LABEL[c]
        mask = test_combos == label
        r = {"scope": label}
        r.update(scope_metrics(test_s[mask], test_y[mask], thresholds))
        rows.append(r)

    # --- Write metrics CSV ---
    fields = ["scope", "n_pairs", "n_pos", "n_neg",
              "pos_sim_mean", "neg_sim_mean", "auc",
              "auc_ci95_low", "auc_ci95_high", "eer"]
    for far in FAR_TARGETS:
        tag = f"far{far:g}"
        fields += [f"tar@{tag}", f"realized_far@{tag}", f"acc@{tag}"]
    csv_path = args.out_dir / "verification_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})

    # --- ROC plot: overall + per pose-bin ---
    plt.figure(figsize=(7.5, 7))
    fpr, tpr, _ = roc_curve(test_y, test_s)
    plt.plot(fpr, tpr, color="black", lw=2.5,
             label=f"overall (AUC={overall['auc']:.3f})")
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(COMBOS):
        label = COMBO_LABEL[c]
        mask = test_combos == label
        if len(np.unique(test_y[mask])) < 2:
            continue
        f, t, _ = roc_curve(test_y[mask], test_s[mask])
        a = roc_auc_score(test_y[mask], test_s[mask])
        plt.plot(f, t, lw=1.6, color=cmap(i / (len(COMBOS) - 1)),
                 label=f"{label} (AUC={a:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xscale("log")
    plt.xlim(1e-3, 1)
    plt.xlabel("False Accept Rate (log)")
    plt.ylabel("True Accept Rate")
    plt.title("Face verification ROC — overall and by pose-bin combination (test)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    roc_path = args.out_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # --- Similarity-by-pose plot (the pose effect the saturated AUC hides) ---
    plt.figure(figsize=(9, 6))
    pos_data, neg_data, xlabels = [], [], []
    for c in COMBOS:
        label = COMBO_LABEL[c]
        mask = test_combos == label
        pos_data.append(test_s[mask][test_y[mask] == 1])
        neg_data.append(test_s[mask][test_y[mask] == 0])
        xlabels.append(label.replace("/", "\n/"))
    positions = np.arange(len(COMBOS))
    bp_pos = plt.boxplot(pos_data, positions=positions - 0.18, widths=0.32,
                         patch_artist=True, showfliers=True)
    bp_neg = plt.boxplot(neg_data, positions=positions + 0.18, widths=0.32,
                         patch_artist=True, showfliers=False)
    for b in bp_pos["boxes"]:
        b.set_facecolor("#4c72b0"); b.set_alpha(0.85)
    for b in bp_neg["boxes"]:
        b.set_facecolor("#dd8452"); b.set_alpha(0.85)
    for thr in sorted(set(round(t, 4) for t in thresholds.values())):
        fars = ", ".join(f"{f:g}" for f, t in thresholds.items() if round(t, 4) == thr)
        plt.axhline(thr, ls="--", lw=1, color="grey")
        plt.text(-0.45, thr + 0.015, f"threshold={thr:.2f} (FAR={fars})",
                 fontsize=7, ha="left", color="grey")
    plt.xticks(positions, xlabels, fontsize=8)
    plt.ylabel("cosine similarity")
    plt.title("Cosine similarity by pose-bin: positives (blue) vs negatives (orange), test")
    plt.plot([], [], color="#4c72b0", lw=6, label="same identity (positive)")
    plt.plot([], [], color="#dd8452", lw=6, label="different identity (negative)")
    plt.legend(loc="center right", fontsize=8)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    sim_path = args.out_dir / "similarity_by_pose.png"
    plt.savefig(sim_path, dpi=150)
    plt.close()

    # --- Console report ---
    print("\n=== TEST metrics (frozen val thresholds) ===")
    hdr = f"{'scope':<24}{'n_pos':>6}{'n_neg':>6}{'AUC':>8}{'EER':>7}" \
          f"{'TAR@1e-3':>10}{'TAR@1e-2':>10}{'acc@1e-2':>10}"
    print(hdr)
    for r in rows:
        print(f"{r['scope']:<24}{r['n_pos']:>6}{r['n_neg']:>6}"
              f"{r['auc']:>8.3f}{r['eer']:>7.3f}"
              f"{r['tar@far0.001']:>10.3f}{r['tar@far0.01']:>10.3f}{r['acc@far0.01']:>10.3f}")
    pp = next(r for r in rows if r["scope"] == "profile/profile")
    print(f"\nprofile/profile AUC 95% CI (1000x bootstrap): "
          f"[{pp['auc_ci95_low']:.3f}, {pp['auc_ci95_high']:.3f}]")
    print("\nMean positive (same-identity) cosine by pose-bin — the pose effect:")
    for r in rows:
        if r["scope"] != "overall":
            print(f"  {r['scope']:<26} pos_sim={r['pos_sim_mean']:.3f}  neg_sim={r['neg_sim_mean']:.3f}")
    print(f"\nSaved: {csv_path}\n       {roc_path}\n       {sim_path}")


if __name__ == "__main__":
    main()
