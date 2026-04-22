"""Face detection and embedding using InsightFace.

Uses the buffalo_l analysis model which provides:
  - Face detection (RetinaFace)
  - 512-d ArcFace embeddings (good for clustering / recognition)

The model weights are downloaded on first run (~300 MB) and cached.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

# InsightFace's face_align module emits a FutureWarning every face it
# aligns because it uses scikit-image's old SimilarityTransform API.
# Third-party code, harmless. Silence it so the index log doesn't fill
# with the same warning tens of thousands of times.
warnings.filterwarnings(
    "ignore",
    message=".*`estimate` is deprecated.*",
    category=FutureWarning,
)

log = logging.getLogger(__name__)

_app = None


@dataclass
class FaceResult:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    det_score: float
    embedding: np.ndarray  # shape (512,), float32


def _load() -> object:
    global _app
    if _app is not None:
        return _app

    from insightface.app import FaceAnalysis

    log.info("loading InsightFace buffalo_l")
    _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def detect_and_embed(img: np.ndarray) -> list[FaceResult]:
    """Detect faces in an RGB image and return bboxes + embeddings.

    *img* must be RGB uint8 (H, W, 3).
    """
    app = _load()
    # InsightFace expects BGR
    img_bgr = img[:, :, ::-1]
    faces = app.get(img_bgr)

    results: list[FaceResult] = []
    for face in faces:
        bbox = tuple(float(v) for v in face.bbox)
        results.append(FaceResult(
            bbox=bbox,
            det_score=float(face.det_score),
            embedding=face.embedding.astype(np.float32),
        ))
    return results
