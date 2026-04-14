"""YOLO object detection.

Loads YOLOv8n (nano) by default — fast on MPS, 80 COCO classes.
The model is cached in the user's cache dir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_model = None


@dataclass
class Detection:
    label: str
    conf: float
    x: float  # normalized center x
    y: float
    w: float
    h: float


def _load(model_name: str = "yolo11n.pt") -> object:
    global _model
    if _model is not None:
        return _model

    from ultralytics import YOLO
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("loading YOLO %s on %s", model_name, device)
    _model = YOLO(model_name)
    _model.to(device)
    return _model


def detect(
    img: np.ndarray,
    conf_threshold: float = 0.3,
    model_name: str = "yolo11n.pt",
) -> list[Detection]:
    """Run YOLO on an RGB image array and return detections."""
    model = _load(model_name)
    results = model(img, verbose=False, conf=conf_threshold)

    dets: list[Detection] = []
    for r in results:
        h_img, w_img = r.orig_shape
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            # xyxy → normalized center + wh
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            dets.append(Detection(
                label=label,
                conf=conf,
                x=(x1 + x2) / 2 / w_img,
                y=(y1 + y2) / 2 / h_img,
                w=(x2 - x1) / w_img,
                h=(y2 - y1) / h_img,
            ))
    return dets
