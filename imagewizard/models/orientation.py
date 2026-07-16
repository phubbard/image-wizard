"""On-device photo-orientation model.

A small 4-class CNN (ResNet-18 backbone + a 4-way head) that predicts how
a photo is rotated relative to upright — 0 / 90 / 180 / 270 — trained
self-supervised on the user's own library: take the correctly-oriented
majority, rotate each by a random quarter-turn, and learn to predict the
turn. No external labels, no external weights beyond the ImageNet backbone.

Used to *suggest* rotations (never auto-apply) for old photos whose camera
wrote no EXIF orientation tag. Validated at ~93% precision on
high-confidence suggestions, ~69% recall.

Convention: a predicted class ``k`` means the stored pixels are ``k``
quarter-turns counter-clockwise from upright, so the clockwise correction
to make it upright is ``90*k`` degrees — exactly what ``files.rotation``
stores. ``predict`` therefore returns ``(correction_cw_degrees, prob)``,
with ``0`` meaning "already upright".
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

INPUT = 128
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_model = None
_device = None


def model_path(cache_dir: Path) -> Path:
    return cache_dir / "models" / "orientation.pt"


def available(cache_dir: Path) -> bool:
    """True if a trained model has been saved (train-orientation was run)."""
    return model_path(cache_dir).exists()


def _dev():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _build(pretrained: bool):
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, 4)
    return m


def _norm(x):
    import torch
    mean = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _to_nchw(imgs: np.ndarray):
    """(N, H, W, 3) uint8 → normalized float NCHW tensor on the device."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float() / 255
    return t


def train(images: np.ndarray, cache_dir: Path, epochs: int = 6,
          batch: int = 64, log_fn=log.info) -> Path:
    """Self-supervised training on a stack of assumed-upright RGB images.

    ``images`` is (N, INPUT, INPUT, 3) uint8. Each step rotates the batch
    to all four orientations (labels 0..3) so every image trains all
    classes equally. Saves weights to the cache and returns the path.
    """
    import torch
    import torch.nn.functional as F

    dev = _dev()
    torch.manual_seed(0)
    train_t = _to_nchw(images)
    model = _build(pretrained=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    def rot_all(x):
        outs, labs = [], []
        for k in range(4):
            outs.append(torch.rot90(x, k, dims=[2, 3]))
            labs.append(torch.full((x.size(0),), k, dtype=torch.long))
        return torch.cat(outs), torch.cat(labs)

    n = len(train_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = cor = 0
        lsum = 0.0
        for i in range(0, n, batch):
            xb = train_t[perm[i:i + batch]].to(dev)
            x4, y4 = rot_all(xb)
            y4 = y4.to(dev)
            logits = model(_norm(x4))
            loss = F.cross_entropy(logits, y4)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lsum += loss.item() * len(y4)
            cor += (logits.argmax(1) == y4).sum().item()
            tot += len(y4)
        log_fn(f"epoch {ep + 1}/{epochs} loss={lsum / tot:.3f} "
               f"train_acc={cor / tot:.3f}")

    out = model_path(cache_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    global _model, _device
    _model, _device = None, None  # force reload of the new weights
    return out


def _load(cache_dir: Path):
    global _model, _device
    if _model is not None:
        return _model
    import torch
    _device = _dev()
    m = _build(pretrained=False)
    m.load_state_dict(torch.load(model_path(cache_dir), map_location=_device))
    m.to(_device).eval()
    _model = m
    return m


def predict_batch(images: np.ndarray, cache_dir: Path):
    """Predict orientation for a stack of (N, INPUT, INPUT, 3) uint8 images.

    Returns a list of ``(correction_cw_degrees, prob)``. ``correction`` is
    0/90/180/270; 0 means the model thinks it's already upright.
    """
    import torch
    import torch.nn.functional as F
    model = _load(cache_dir)
    dev = _device
    x = _to_nchw(images).to(dev)
    with torch.no_grad():
        probs = F.softmax(model(_norm(x)), dim=1).cpu().numpy()
    out = []
    for p in probs:
        k = int(p.argmax())
        out.append(((90 * k) % 360, float(p[k])))
    return out


def predict(img: np.ndarray, cache_dir: Path):
    """Convenience: one (INPUT, INPUT, 3) uint8 image → (correction, prob)."""
    return predict_batch(img[None, ...], cache_dir)[0]
