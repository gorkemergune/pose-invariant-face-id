"""Gradio demo: are these two photos the same person?

Upload two photos. For each we run RetinaFace detection + 5-point alignment
(112x112), extract a 512-d ArcFace embedding (InsightFace ``buffalo_l``), and
compute the cosine similarity between them. The decision uses the frozen
verification threshold selected on the validation split in Step 6
(``evaluate.py``): same person if cosine >= 0.44.

Run:
    python app.py
Then open the printed local URL in a browser.
"""

from __future__ import annotations

import os.path as osp

import cv2
import numpy as np
import gradio as gr

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align

# Frozen operating point from Step 6 (val-selected threshold for FAR=1e-3/1e-2).
THRESHOLD = 0.44

_det_app = None
_rec_model = None


def _load_models():
    """Lazy-load detection + recognition models once."""
    global _det_app, _rec_model
    if _det_app is None:
        _det_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        _det_app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.3)
    if _rec_model is None:
        root = osp.expanduser("~/.insightface/models/buffalo_l")
        _rec_model = get_model(osp.join(root, "w600k_r50.onnx"))
        _rec_model.prepare(ctx_id=-1)
    return _det_app, _rec_model


def embed(image_rgb: np.ndarray):
    """Detect the most confident face, align, and return a unit-norm embedding."""
    det, rec = _load_models()
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = det.get(img_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: float(f.det_score))
    aligned = face_align.norm_crop(img_bgr, landmark=face.kps, image_size=112)
    vec = np.asarray(rec.get_feat(aligned), dtype=np.float32).ravel()
    vec /= np.linalg.norm(vec) + 1e-12
    return vec


def compare(img1, img2):
    if img1 is None or img2 is None:
        return "Please upload two images."
    e1, e2 = embed(img1), embed(img2)
    if e1 is None or e2 is None:
        which = []
        if e1 is None:
            which.append("image 1")
        if e2 is None:
            which.append("image 2")
        return f"No face detected in {', '.join(which)}. Try a clearer / less extreme-pose photo."
    sim = float(np.dot(e1, e2))
    same = sim >= THRESHOLD
    verdict = "✅ SAME person" if same else "❌ DIFFERENT people"
    return (f"{verdict}\n\n"
            f"cosine similarity : {sim:.3f}\n"
            f"decision threshold : {THRESHOLD:.2f}  (frozen from validation, Step 6)\n"
            f"margin             : {sim - THRESHOLD:+.3f}")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pose-Invariant Face Verification") as demo:
        gr.Markdown(
            "# Pose-Invariant Face Verification\n"
            "Upload two photos — the model reports whether they are the **same "
            "person**, using ArcFace embeddings and a cosine-similarity threshold "
            f"of **{THRESHOLD:.2f}** frozen from the validation split. Faces may be "
            "frontal, half-profile, or profile."
        )
        with gr.Row():
            in1 = gr.Image(label="Photo 1", type="numpy", sources=["upload", "webcam"])
            in2 = gr.Image(label="Photo 2", type="numpy", sources=["upload", "webcam"])
        btn = gr.Button("Compare", variant="primary")
        out = gr.Textbox(label="Result", lines=5)
        btn.click(compare, inputs=[in1, in2], outputs=out)
    return demo


if __name__ == "__main__":
    build_ui().launch()
