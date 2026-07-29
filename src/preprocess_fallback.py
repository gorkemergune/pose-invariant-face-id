"""Fallback detection pass for images that failed the main preprocessing.

Re-runs RetinaFace on only the rows marked ``no_face`` in
``results/preprocess_log.csv`` using a larger detector input and a lower
detection threshold (extreme-pose / profile faces are hard for the frontal-
biased detector at the default settings). Recovered faces are aligned to
112x112, written into ``data/processed/aligned/``, and their rows in the
manifest are updated in place. Rows that still fail are left as ``no_face``.

Usage:
    python src/preprocess_fallback.py
    python src/preprocess_fallback.py --det-size 1024 --det-thresh 0.1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from insightface.utils import face_align
from preprocess import PROJECT_ROOT, build_app, pick_face


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
    p.add_argument("--det-size", type=int, default=1024)
    p.add_argument("--det-thresh", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=112)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with args.log.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    targets = [r for r in rows if r["status"] != "ok"]
    if not targets:
        print("No failed rows to retry — nothing to do.")
        return

    print(f"Retrying {len(targets)} failed image(s) with "
          f"det_size={args.det_size}, det_thresh={args.det_thresh} ...")
    app = build_app(args.det_size, args.det_thresh)

    recovered = 0
    still_failing = []

    for row in targets:
        path = args.raw_dir / row["filename"]
        img = cv2.imread(str(path))
        if img is None:
            row["status"] = "read_error"
            still_failing.append(row["filename"])
            continue

        faces = app.get(img)
        row["num_faces"] = len(faces)
        face = pick_face(faces)

        if face is None:
            row["status"] = "no_face"
            still_failing.append(row["filename"])
            print(f"  still no face: {row['filename']}")
            continue

        aligned = face_align.norm_crop(img, landmark=face.kps, image_size=args.image_size)
        out_path = args.out_dir / row["filename"]
        cv2.imwrite(str(out_path), aligned)

        row["status"] = "ok"
        row["det_score"] = round(float(face.det_score), 4)
        row["bbox"] = " ".join(f"{v:.1f}" for v in face.bbox.tolist())
        row["aligned_path"] = str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        recovered += 1
        print(f"  recovered: {row['filename']}  (det_score={row['det_score']})")

    with args.log.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    print("\n=== Fallback summary ===")
    print(f"Retried        : {len(targets)}")
    print(f"Recovered      : {recovered}")
    print(f"Still failing  : {len(still_failing)}")
    for f in still_failing:
        print(f"    {f}")
    print(f"Total aligned  : {n_ok} / {len(rows)}")


if __name__ == "__main__":
    main()
