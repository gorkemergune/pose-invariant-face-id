"""Extract 512-d ArcFace embeddings for the aligned FEI crops.

Uses the pretrained ArcFace recognition model from InsightFace's ``buffalo_l``
pack (``w600k_r50.onnx``) directly on the 112x112 aligned crops produced by
``preprocess.py``. Embeddings are the raw (non-unit-norm) 512-d model outputs;
downstream cosine-similarity code L2-normalizes them.

Output ``data/processed/embeddings.npz`` (git-ignored — biometric-derived data):
    image_paths : (N,)   str    relative aligned-crop paths
    embeddings  : (N,512) float32
    subject_ids : (N,)   int
    pose_bins   : (N,)   str
    model_name  : ()     str    e.g. "arcface_buffalo_l_w600k_r50"

AdaFace note: a second AdaFace embedding was intended for comparison, but AdaFace
ships no packaged pip distribution and its pretrained weights are only available
via a Google Drive download, which is fragile to automate here. It is therefore
skipped; ArcFace is used as the single embedding model. To add it later, load an
AdaFace checkpoint and write a parallel ``embeddings_adaface.npz`` with the same
layout.

Usage:
    python src/embed.py
    python src/embed.py --batch-size 128
"""

from __future__ import annotations

import argparse
import csv
import os.path as osp
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from insightface.model_zoo import get_model

FNAME_RE = re.compile(r"^(?P<subject>\d+)-(?P<img>\d+)\.jpg$", re.IGNORECASE)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "arcface_buffalo_l_w600k_r50"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aligned-dir", type=Path,
                   default=PROJECT_ROOT / "data" / "processed" / "aligned")
    p.add_argument("--pose-labels", type=Path,
                   default=PROJECT_ROOT / "results" / "pose_labels.csv")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data" / "processed" / "embeddings.npz")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def load_pose_bins(pose_labels: Path) -> dict[str, str]:
    bins = {}
    if pose_labels.exists():
        with pose_labels.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["status"] == "ok":
                    bins[r["filename"]] = r["pose_bin"]
    return bins


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(args.aligned_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No aligned crops found in {args.aligned_dir}")
    pose_bins = load_pose_bins(args.pose_labels)

    root = osp.expanduser("~/.insightface/models/buffalo_l")
    rec = get_model(osp.join(root, "w600k_r50.onnx"))
    rec.prepare(ctx_id=-1)
    print(f"Loaded ArcFace recognition model ({MODEL_NAME}); "
          f"embedding {len(images)} aligned crops ...")

    paths, subjects, bins, feats = [], [], [], []

    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        out = np.asarray(rec.get_feat(batch_imgs), dtype=np.float32)
        for (rel, subj, b), vec in zip(batch_meta, out):
            paths.append(rel)
            subjects.append(subj)
            bins.append(b)
            feats.append(vec)
        batch_imgs.clear()
        batch_meta.clear()

    for path in tqdm(images, desc="Embedding", unit="img"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        m = FNAME_RE.match(path.name)
        subj = int(m.group("subject")) if m else -1
        rel = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        batch_imgs.append(img)
        batch_meta.append((rel, subj, pose_bins.get(path.name, "")))
        if len(batch_imgs) >= args.batch_size:
            flush()
    flush()

    embeddings = np.vstack(feats).astype(np.float32)
    np.savez_compressed(
        args.out,
        image_paths=np.array(paths),
        embeddings=embeddings,
        subject_ids=np.array(subjects, dtype=np.int32),
        pose_bins=np.array(bins),
        model_name=np.array(MODEL_NAME),
    )

    print("\n=== Embedding summary ===")
    print(f"Aligned crops found : {len(images)}")
    print(f"Embeddings extracted : {embeddings.shape[0]}  (dim={embeddings.shape[1]})")
    print(f"Model                : {MODEL_NAME}")
    print(f"Saved                -> {args.out}  (git-ignored)")


if __name__ == "__main__":
    main()
