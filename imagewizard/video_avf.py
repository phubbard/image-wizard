"""AVFoundation (VideoToolbox) video frame extraction — macOS only.

Why this exists: OpenCV's bundled FFmpeg on recent builds can't open the
HEVC ``.mov``/``.mp4`` that modern iPhones and GoPros produce, and it
never applies the container's display-rotation matrix — so portrait
videos came out sideways (or not at all). AVFoundation decodes HEVC in
hardware and, with ``appliesPreferredTrackTransform``, hands back an
already-upright frame that matches what Photos / QuickTime show. No
rotation math, no codec worries.

The whole thing is import-guarded: if the pyobjc frameworks aren't present
(non-macOS, or a slim install), ``available()`` returns False and callers
fall back to OpenCV.
"""
from __future__ import annotations

import io
import math
import threading
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

_AVAILABLE: bool | None = None


def available() -> bool:
    """True if the AVFoundation pyobjc frameworks can be imported."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import AVFoundation  # noqa: F401
            import CoreMedia  # noqa: F401
            import AppKit  # noqa: F401
            from Foundation import NSURL  # noqa: F401
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def extract_poster(
    path: Path, ts_sec: float = 1.0, timeout: float = 30.0
) -> tuple[np.ndarray, float | None] | None:
    """Decode one upright frame near ``ts_sec``.

    Returns ``(rgb_uint8_HxWx3, duration_sec_or_None)`` or ``None`` if
    AVFoundation is unavailable or can't produce a frame (caller falls
    back to OpenCV). The frame already has the track's preferred
    transform applied, so portrait video comes back portrait.
    """
    if not available():
        return None
    import AVFoundation
    import CoreMedia
    import AppKit
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(path))
    asset = AVFoundation.AVURLAsset.URLAssetWithURL_options_(url, None)

    dur = CoreMedia.CMTimeGetSeconds(asset.duration())
    if not (isinstance(dur, float) and math.isfinite(dur) and dur > 0):
        dur = None

    gen = AVFoundation.AVAssetImageGenerator.assetImageGeneratorWithAsset_(asset)
    gen.setAppliesPreferredTrackTransform_(True)
    seek = min(ts_sec, dur / 2) if dur else ts_sec
    seek = max(seek, 0.0)
    t = CoreMedia.CMTimeMakeWithSeconds(seek, 600)

    done = threading.Event()
    box: dict = {}

    def handler(image, actual_time, error):
        try:
            if image is not None:
                rep = AppKit.NSBitmapImageRep.alloc().initWithCGImage_(image)
                data = rep.TIFFRepresentation()
                box["arr"] = np.asarray(
                    Image.open(io.BytesIO(bytes(data))).convert("RGB"))
            else:
                box["err"] = str(error)
        except Exception as e:  # pragma: no cover - defensive
            box["err"] = repr(e)
        finally:
            done.set()

    with warnings.catch_warnings():
        # pyobjc emits an ObjCPointerWarning for the CGImageRef arg; benign.
        warnings.simplefilter("ignore")
        gen.generateCGImageAsynchronouslyForTime_completionHandler_(t, handler)
        if not done.wait(timeout):
            return None

    arr = box.get("arr")
    if arr is None:
        return None
    return arr, dur
