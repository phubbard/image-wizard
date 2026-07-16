"""Orientation model plumbing — shapes and the correction convention.

The model's *accuracy* is validated empirically (not in CI, since it needs
training), but the class→degrees mapping is pure and must be right: a
predicted class k means the pixels are k quarter-turns CCW from upright, so
the clockwise correction stored in files.rotation is 90*k.
"""
from __future__ import annotations

import numpy as np
import pytest

from imagewizard.models import orientation as o


def test_to_nchw_shape_and_range():
    pytest.importorskip("torch")
    imgs = np.full((3, o.INPUT, o.INPUT, 3), 255, np.uint8)
    t = o._to_nchw(imgs)
    assert tuple(t.shape) == (3, 3, o.INPUT, o.INPUT)
    assert float(t.max()) <= 1.0 and float(t.min()) >= 0.0


@pytest.mark.parametrize("k,want_deg", [(0, 0), (1, 90), (2, 180), (3, 270)])
def test_correction_convention(k, want_deg, tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class Stub(nn.Module):
        def __init__(self, cls):
            super().__init__()
            self.cls = cls

        def forward(self, x):
            logits = torch.zeros(x.shape[0], 4)
            logits[:, self.cls] = 10.0
            return logits

    o._model = Stub(k)
    o._device = torch.device("cpu")
    try:
        imgs = np.zeros((1, o.INPUT, o.INPUT, 3), np.uint8)
        deg, prob = o.predict_batch(imgs, tmp_path)[0]
        assert deg == want_deg
        assert deg in (0, 90, 180, 270)
        assert 0.9 < prob <= 1.0
    finally:
        o._model = None
        o._device = None
