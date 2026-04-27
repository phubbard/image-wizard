"""Video frame extraction.

V1 of video support treated each video as a single frame (poster at
~1s) and ran the standard ML stages on it. V2 keeps that capability
(``extract_poster``) and adds multi-frame sampling via
``frame_schedule`` + ``iter_frames``: the pipeline samples N frames
per video, runs ML on each, and stores per-frame detections / faces /
CLIP embeddings keyed by ``frames.id``.

Sampling default: 1 fps for the first 60 seconds, then 1 frame every
10 seconds. Deterministic — re-indexing produces the same timestamps
so on-disk thumb caches and frame rows stay valid.

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
from typing import Iterator

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Where in the video to grab the poster. 1 second is enough to skip
# the typical fade-in / black frames at the very start, but short
# enough that even a 2-second clip has something to show.
DEFAULT_POSTER_SEC = 1.0

# V2 default sampling schedule.
DENSE_FPS_SECONDS = 60.0    # cover this much at 1 fps
DENSE_FPS = 1.0
SPARSE_INTERVAL = 10.0      # then one frame per this many seconds
MAX_FRAMES_PER_VIDEO = 600  # safety cap; a 100-min clip hits this


def frame_schedule(duration_sec: float | None) -> list[float]:
    """Return the ``ts_sec`` list at which to sample frames from a video.

    Deterministic so the same video always yields the same set of
    sampled timestamps — important for incremental re-runs to be
    no-ops on already-processed videos.
    """
    if not duration_sec or duration_sec <= 0:
        return [DEFAULT_POSTER_SEC]
    out: list[float] = []
    # Phase 1: dense sampling near the start.
    t = 0.0
    while t < min(duration_sec, DENSE_FPS_SECONDS):
        out.append(round(t, 3))
        t += 1.0 / DENSE_FPS
    # Phase 2: sparse sampling for the rest.
    t = max(t, DENSE_FPS_SECONDS)
    while t < duration_sec:
        out.append(round(t, 3))
        t += SPARSE_INTERVAL
    if len(out) > MAX_FRAMES_PER_VIDEO:
        # Keep the dense start, sparsen the tail by even subsample.
        head = out[:MAX_FRAMES_PER_VIDEO // 2]
        tail = out[MAX_FRAMES_PER_VIDEO // 2:]
        step = max(1, len(tail) // (MAX_FRAMES_PER_VIDEO - len(head)))
        out = head + tail[::step]
        out = out[:MAX_FRAMES_PER_VIDEO]
    return out


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


def _downscale(rgb: np.ndarray, max_pixels: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h * w <= max_pixels:
        return rgb
    scale = (max_pixels / (h * w)) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil)


def iter_frames(
    path: Path,
    timestamps: list[float] | None = None,
    max_pixels: int = 4096 * 4096,
) -> Iterator[tuple[float, np.ndarray, float | None]]:
    """Yield ``(actual_ts_sec, rgb_uint8, duration_sec)`` for each
    requested timestamp.

    If ``timestamps`` is None, the default schedule is used (computed
    from the video's duration). Frames are yielded in order. The
    actual timestamp returned can differ slightly from the requested
    one — a seek lands at the nearest keyframe and we then decode
    forward to the requested PTS, so jagged-keyframe codecs (older
    H.263, some MJPEG) may snap to the keyframe.

    Generator semantics matter here: the consumer processes one frame
    at a time so memory stays bounded regardless of video length. A
    1-hour video with 414 sampled frames never holds more than one
    frame's RGB array in memory.
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

        # Duration: prefer container-level, fall back to stream-level.
        duration_sec: float | None = None
        if container.duration:
            duration_sec = container.duration / 1_000_000
        elif stream.duration is not None and stream.time_base is not None:
            duration_sec = float(stream.duration * stream.time_base)

        targets = (
            list(timestamps) if timestamps is not None
            else frame_schedule(duration_sec)
        )

        for ts in targets:
            # Bound the seek target so we don't run off the end.
            seek_target = ts
            if duration_sec is not None and duration_sec > 0:
                seek_target = min(seek_target, duration_sec - 0.05)
            seek_target = max(seek_target, 0.0)

            try:
                container.seek(int(seek_target * 1_000_000), any_frame=False)
            except av.AVError:
                log.debug("seek failed for %s @ %.2fs", path, seek_target)
                continue

            frame = next(iter(container.decode(stream)), None)
            if frame is None:
                continue
            actual = float(
                frame.pts * stream.time_base
                if frame.pts is not None and stream.time_base is not None
                else seek_target
            )
            rgb = frame.to_ndarray(format="rgb24")
            rgb = _downscale(rgb, max_pixels)
            yield actual, rgb, duration_sec
