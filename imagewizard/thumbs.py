"""On-disk thumbnail cache.

Thumbnails are 512px (longest edge) JPEGs stored in the cache directory,
organized by the first two characters of the content hash for filesystem
friendliness.

    ~/Library/Caches/image-wizard/thumbs/ab/ab3fe...jpg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

THUMB_SIZE = 512
THUMB_QUALITY = 80


def thumb_path(cache_dir: Path, content_hash: str) -> Path:
    return cache_dir / "thumbs" / content_hash[:2] / f"{content_hash}.jpg"


def ensure_thumbnail(
    img: np.ndarray,
    cache_dir: Path,
    content_hash: str,
) -> Path:
    """Write a thumbnail if it doesn't exist. Return the path."""
    out = thumb_path(cache_dir, content_hash)
    if out.exists():
        return out

    pil = Image.fromarray(img)
    pil.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out, "JPEG", quality=THUMB_QUALITY)
    return out
