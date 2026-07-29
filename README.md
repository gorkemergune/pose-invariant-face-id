# pose-invariant-face-id

Pose-Invariant Face Verification: an identity-disjoint train/test face
verification system built on pretrained ArcFace embeddings, with an analysis of
how verification accuracy degrades as head pose (yaw) moves from frontal to
profile.

## Dataset

This project uses the **FEI Face Database** — 200 subjects × 14 images = 2800
color images, captured across roughly 180° of head rotation (frontal, half
profile, profile), with some illumination variation. Each file is named
`<subjectID>-<imageNum>.jpg` (subject IDs 1–200, image numbers 01–14).

Only the `originalimages_*` archives (the full pose range) are used for the main
pipeline. The `frontalimages_*` and `*_averagefaceimages` archives are optional
reference material and are not part of the pipeline.

> **Citation.** FEI Face Database, Artificial Intelligence Laboratory of FEI,
> São Bernardo do Campo, São Paulo, Brazil.
> Thomaz, C. E. and Giraldi, G. A., *A new ranking method for principal
> components analysis and its application to face image analysis*, Image and
> Vision Computing, 28(6):902-913, 2010.
> Project page: https://fei.edu.br/~cet/facedatabase.html

**Licensing note.** The `LICENSE` file in this repo (Apache 2.0) covers the
*code*. The FEI images themselves are governed by the FEI Face Database's own
terms (research/academic use); raw and processed images are **not** redistributed
in this repository (see `.gitignore`).

## Project structure

```
data/
  raw/            # extracted FEI original images (git-ignored)
  processed/      # aligned faces, embeddings, labels, splits (git-ignored)
src/
  preprocess.py       # face detection + landmark alignment (112x112)
  pose_estimation.py  # per-image yaw estimation + pose binning
  split.py            # identity-disjoint train/val/test split + pairs
  embed.py            # ArcFace (buffalo_l) 512-d embedding extraction
  evaluate.py         # ROC/AUC, EER, TAR@FAR, pose-binned breakdown
  visualize.py        # UMAP/t-SNE embedding scatter
  retrieval.py        # (optional) FAISS retrieval demo
notebooks/
results/          # logs, tables, plots
models/           # cached / fine-tuned weights (git-ignored)
app.py            # (optional) Gradio same-person demo
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   *nix: source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

1. **Data prep** — extract `originalimages_part1-4.zip` into `data/raw/`
   (200×14 = 2800 images verified).
2. **Preprocess** — `python src/preprocess.py`: RetinaFace detection + 5-point
   landmarks, aligned to 112×112 (ArcFace standard), saved to
   `data/processed/aligned/`. Failures logged to `results/preprocess_log.csv`.
3. **Pose labeling** — per-image yaw → bins: frontal `<20°`, half `20–60°`,
   profile `>60°`.
4. **Identity-disjoint split** — ~160 train / 20 val / 20 test subjects.
5. **Embeddings** — pretrained ArcFace (`buffalo_l`), 512-d.
6. **Evaluation** — verification metrics, overall and per pose-bin pair.
7. **Visualization** — UMAP/t-SNE of embeddings by identity and pose.

## Results

_To be filled in as the pipeline runs._
