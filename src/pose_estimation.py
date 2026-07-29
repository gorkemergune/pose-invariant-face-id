"""Per-image head-pose (yaw) estimation and pose binning for the FEI dataset.

The FEI images carry no explicit yaw-angle labels, so we estimate one with a
pretrained head-pose model. We use InsightFace's ``landmark_3d_68`` model (part
of the ``buffalo_l`` pack already downloaded during preprocessing), which fits
3D facial landmarks and returns a ``pose = [pitch, yaw, roll]`` triple per
detected face. Pose is estimated on the *raw* images, where the true head
rotation is intact (alignment would remove in-plane roll but we want the
original geometry).

Each image is binned by absolute yaw:
    frontal      |yaw| < 20 deg
    half-profile 20 <= |yaw| <= 60 deg
    profile      |yaw| > 60 deg

Output: ``results/pose_labels.csv``.

Usage:
    python src/pose_estimation.py
    python src/pose_estimation.py --limit 50
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
from tqdm import tqdm

from insightface.app import FaceAnalysis

FNAME_RE = re.compile(r"^(?P<subject>\d+)-(?P<img>\d+)\.jpg$", re.IGNORECASE)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRONTAL_MAX = 20.0   # |yaw| < 20      -> frontal
HALF_MAX = 60.0      # 20 <= |yaw| <=60 -> half-profile ; >60 -> profile


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "pose_labels.csv",
    )
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--det-thresh", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def pose_bin(abs_yaw: float) -> str:
    if abs_yaw < FRONTAL_MAX:
        return "frontal"
    if abs_yaw <= HALF_MAX:
        return "half-profile"
    return "profile"


def build_app(det_size: int, det_thresh: float) -> FaceAnalysis:
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size), det_thresh=det_thresh)
    return app


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(args.raw_dir.glob("*.jpg"))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No .jpg images found in {args.raw_dir}")

    print(f"Loading detector + landmark_3d_68 pose model (det_size={args.det_size}) ...")
    app = build_app(args.det_size, args.det_thresh)

    rows = []
    bin_counts = {"frontal": 0, "half-profile": 0, "profile": 0}
    n_ok = n_fail = 0

    for path in tqdm(images, desc="Estimating pose", unit="img"):
        m = FNAME_RE.match(path.name)
        subject = int(m.group("subject")) if m else ""
        img_num = int(m.group("img")) if m else ""
        row = {
            "filename": path.name,
            "subject_id": subject,
            "image_num": img_num,
            "status": "",
            "yaw": "",
            "pitch": "",
            "roll": "",
            "abs_yaw": "",
            "pose_bin": "",
            "det_score": "",
        }

        img = cv2.imread(str(path))
        if img is None:
            row["status"] = "read_error"
            rows.append(row)
            n_fail += 1
            continue

        faces = app.get(img)
        if not faces:
            row["status"] = "no_face"
            rows.append(row)
            n_fail += 1
            continue

        face = max(faces, key=lambda f: float(f.det_score))
        pitch, yaw, roll = (float(v) for v in face.pose)
        abs_yaw = abs(yaw)
        b = pose_bin(abs_yaw)

        row.update(
            status="ok",
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2),
            abs_yaw=round(abs_yaw, 2),
            pose_bin=b,
            det_score=round(float(face.det_score), 4),
        )
        rows.append(row)
        bin_counts[b] += 1
        n_ok += 1

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Pose labeling summary ===")
    print(f"Total images      : {len(images)}")
    print(f"Pose estimated    : {n_ok}")
    print(f"No detection      : {n_fail}")
    print("Pose-bin distribution (of estimated):")
    for b in ("frontal", "half-profile", "profile"):
        c = bin_counts[b]
        pct = 100.0 * c / n_ok if n_ok else 0.0
        print(f"  {b:<13}: {c:5d}  ({pct:4.1f}%)")
    print(f"Labels -> {args.out}")


if __name__ == "__main__":
    main()
