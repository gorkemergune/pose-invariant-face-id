"""Face detection + landmark alignment for the FEI dataset.

Runs RetinaFace (via InsightFace's ``buffalo_l`` detection model) over every
image in ``data/raw/``, aligns each detected face to a 112x112 crop using the
5-point landmarks (the ArcFace standard), and writes the aligned crop to
``data/processed/aligned/`` with the same filename.

A full per-image manifest is written to ``results/preprocess_log.csv`` with a
``status`` column, so detection failures can be filtered with
``status != ok``.

Usage:
    python src/preprocess.py
    python src/preprocess.py --limit 50          # quick smoke test
    python src/preprocess.py --det-size 1024     # larger detector input
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis
from insightface.utils import face_align

# FEI filenames look like "<subjectID>-<imageNum>.jpg", e.g. "48-01.jpg".
FNAME_RE = re.compile(r"^(?P<subject>\d+)-(?P<img>\d+)\.jpg$", re.IGNORECASE)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "aligned",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=PROJECT_ROOT / "results" / "preprocess_log.csv",
    )
    p.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Square detector input size (larger = better recall on profiles, slower).",
    )
    p.add_argument(
        "--det-thresh",
        type=float,
        default=0.3,
        help="Detection score threshold (lower = more permissive for hard poses).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=112,
        help="Aligned output crop size (ArcFace standard is 112).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images (for a quick smoke test).",
    )
    return p.parse_args()


def build_app(det_size: int, det_thresh: float) -> FaceAnalysis:
    """Load only the RetinaFace detection module on CPU."""
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size), det_thresh=det_thresh)
    return app


def pick_face(faces):
    """Pick the most confident detection (FEI images contain one subject)."""
    if not faces:
        return None
    return max(faces, key=lambda f: float(f.det_score))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(args.raw_dir.glob("*.jpg"))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No .jpg images found in {args.raw_dir}")

    print(f"Loading RetinaFace detector (det_size={args.det_size}, "
          f"det_thresh={args.det_thresh}) ...")
    app = build_app(args.det_size, args.det_thresh)

    n_ok = n_fail = 0
    rows = []

    for path in tqdm(images, desc="Aligning faces", unit="img"):
        m = FNAME_RE.match(path.name)
        subject = int(m.group("subject")) if m else ""
        img_num = int(m.group("img")) if m else ""

        row = {
            "filename": path.name,
            "subject_id": subject,
            "image_num": img_num,
            "status": "",
            "num_faces": 0,
            "det_score": "",
            "bbox": "",
            "aligned_path": "",
        }

        img = cv2.imread(str(path))
        if img is None:
            row["status"] = "read_error"
            rows.append(row)
            n_fail += 1
            continue

        faces = app.get(img)
        row["num_faces"] = len(faces)
        face = pick_face(faces)

        if face is None:
            row["status"] = "no_face"
            rows.append(row)
            n_fail += 1
            continue

        aligned = face_align.norm_crop(img, landmark=face.kps, image_size=args.image_size)
        out_path = args.out_dir / path.name
        cv2.imwrite(str(out_path), aligned)

        row["status"] = "ok"
        row["det_score"] = round(float(face.det_score), 4)
        row["bbox"] = " ".join(f"{v:.1f}" for v in face.bbox.tolist())
        row["aligned_path"] = str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        rows.append(row)
        n_ok += 1

    with args.log.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Preprocessing summary ===")
    print(f"Total images   : {len(images)}")
    print(f"Aligned (ok)   : {n_ok}")
    print(f"Failed         : {n_fail}")
    if n_fail:
        multi = [r for r in rows if r["num_faces"] and r["num_faces"] > 1]
        print(f"  (images with >1 detected face: {len(multi)})")
        print("  Failed files:")
        for r in rows:
            if r["status"] != "ok":
                print(f"    {r['filename']}  [{r['status']}]")
    print(f"Aligned crops  -> {args.out_dir}")
    print(f"Manifest/log   -> {args.log}")


if __name__ == "__main__":
    main()
