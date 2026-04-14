"""Image decoding: path → RGB uint8 numpy array.

Handles JPEG/PNG/TIFF via Pillow, HEIC via pillow-heif, and optionally
RAW formats via rawpy (if installed).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_heif_registered = False


def _ensure_heif() -> None:
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        _heif_registered = True
    except ImportError:
        _heif_registered = True  # don't retry


RAW_EXTS = frozenset({".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef"})


def load_image(path: Path, max_pixels: int = 4096 * 4096) -> np.ndarray:
    """Load an image as RGB uint8 (H, W, 3).

    Large images are downscaled so total pixels ≤ *max_pixels* to keep model
    input sizes reasonable and memory bounded.
    """
    ext = path.suffix.lower()

    if ext in RAW_EXTS:
        try:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB)
        except ImportError:
            raise RuntimeError(
                f"rawpy not installed — cannot decode {path.name}. "
                "Install with: uv pip install rawpy"
            )
    else:
        _ensure_heif()
        pil = Image.open(path)
        pil = pil.convert("RGB")
        rgb = np.array(pil)

    h, w = rgb.shape[:2]
    if h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        pil = Image.fromarray(rgb)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        rgb = np.array(pil)

    return rgb
