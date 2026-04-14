"""CLIP image/text embeddings using open_clip.

Provides two operations:
  1. embed_image(img) → 512-d float32 vector
  2. embed_text(query) → 512-d float32 vector

These go into sqlite-vec for nearest-neighbour text→image search.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)

_model = None
_preprocess = None
_tokenizer = None
_device = None


def _load() -> None:
    global _model, _preprocess, _tokenizer, _device

    if _model is not None:
        return

    import open_clip

    _device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("loading CLIP ViT-B-32 on %s", _device)

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=_device,
    )
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")
    _model.eval()


def embed_image(img: np.ndarray) -> np.ndarray:
    """Embed an RGB uint8 image → (512,) float32 L2-normalized."""
    _load()
    from PIL import Image

    pil_img = Image.fromarray(img)
    tensor = _preprocess(pil_img).unsqueeze(0).to(_device)

    with torch.no_grad(), torch.amp.autocast(_device):
        feat = _model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    """Embed a text query → (512,) float32 L2-normalized."""
    _load()
    tokens = _tokenizer([text]).to(_device)

    with torch.no_grad(), torch.amp.autocast(_device):
        feat = _model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze(0).cpu().numpy().astype(np.float32)
