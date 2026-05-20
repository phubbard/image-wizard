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
    return embed_image_batch([img])[0]


def embed_image_batch(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """Embed a batch of RGB uint8 images → list of (512,) float32 L2-normalized.

    Stacks preprocessed tensors and runs a single forward pass — at batch
    sizes ≥4 this is materially faster than serial inference on MPS.
    """
    if not imgs:
        return []
    _load()
    from PIL import Image

    tensors = [_preprocess(Image.fromarray(im)) for im in imgs]
    batch = torch.stack(tensors).to(_device)

    with torch.no_grad(), torch.amp.autocast(_device):
        feats = _model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    arr = feats.cpu().numpy().astype(np.float32)
    return [arr[i] for i in range(arr.shape[0])]


def embed_text(text: str) -> np.ndarray:
    """Embed a text query → (512,) float32 L2-normalized."""
    _load()
    tokens = _tokenizer([text]).to(_device)

    with torch.no_grad(), torch.amp.autocast(_device):
        feat = _model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.squeeze(0).cpu().numpy().astype(np.float32)
