"""Video poster-frame extraction.

V1 of video support treats each video as a single frame for ML
purposes: pluck a poster frame at ~1 second in (or earlier for short
clips), then run the same YOLO/CLIP/InsightFace stages we run on
photos. The full video file streams to the browser via the existing
`/full/{id}` endpoint with a `<video>` tag for browser-friendly
codecs.

Decoding goes through PyAV (Python bindings to FFmpeg). On Apple
Silicon FFmpeg uses VideoToolbox hardware decode for H.264 / HEVC;
ProRes and other codecs fall back to software but still work.

We don't validate codec compatibility here — that's the web layer's
job (it picks `<video>` vs poster image based on the path's
extension).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Where in the video to grab the poster. 1 second is enough to skip
# the typical fade-in / black frames at the very start, but short
# enough that even a 2-second clip has something to show.
DEFAULT_POSTER_SEC = 1.0


def extract_poster(
    path: Path,
    ts_sec: float = DEFAULT_POSTER_SEC,
    max_pixels: int = 4096 * 4096,
) -> tuple[np.ndarray, float | None]:
    """Decode a single frame near ``ts_sec`` plus the video's duration.

    Returns ``(rgb_uint8_HxWx3, duration_sec_or_None)``. Raises on
    unreadable video. Downscales the frame so total pixels ≤ ``max_pixels``
    to keep ML model input bounded — same convention as ``decode.load_image``.

    If the video is shorter than ``ts_sec`` we walk back to the midpoint
    so we never fall off the end.
    """
    try:
        import av
    except ImportError as e:
        raise RuntimeError(
            "PyAV not installed — video support requires `uv pip install av`."
        ) from e

    with av.open(str(path)) as container:
        stream = next(
            (s for s in container.streams if s.type == "video"), None
        )
        if stream is None:
            raise RuntimeError(f"no video stream in {path.name}")

        # Duration: prefer container-level value (more reliable across
        # codecs); fall back to the stream's duration in time_base units.
        duration_sec: float | None = None
        if container.duration:
            duration_sec = container.duration / 1_000_000  # AV_TIME_BASE
        elif stream.duration is not None and stream.time_base is not None:
            duration_sec = float(stream.duration * stream.time_base)

        # Pick a safe seek target: requested ts, but bounded to the
        # midpoint so very short clips don't seek past the end.
        seek_target = ts_sec
        if duration_sec is not None and duration_sec > 0:
            seek_target = min(ts_sec, duration_sec / 2)
        seek_target = max(seek_target, 0.0)

        # Seek by container PTS (AV_TIME_BASE = microseconds).
        try:
            container.seek(int(seek_target * 1_000_000), any_frame=False)
        except av.AVError:
            # Some containers refuse the seek; fall back to first frame.
            log.debug("seek failed for %s — using first frame", path)

        frame = next(iter(container.decode(stream)), None)
        if frame is None:
            raise RuntimeError(f"no decodable frame in {path.name}")

        # Convert the AVFrame to a contiguous RGB uint8 numpy array.
        rgb = frame.to_ndarray(format="rgb24")

    # Match the still-image pipeline's max_pixels behaviour.
    h, w = rgb.shape[:2]
    if h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        rgb = np.array(pil)

    return rgb, duration_sec
