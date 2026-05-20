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

Decoding goes through OpenCV's VideoCapture, which is FFmpeg under
the hood — the same FFmpeg ultralytics already ships in cv2's
.dylibs. Using cv2 here (rather than a second FFmpeg bundle via
pyav) avoids the macOS ObjC class-duplication warning for
``AVFFrameReceiver`` / ``libavdevice``, and keeps the install ~30 MB
smaller.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
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
    t = 0.0
    while t < min(duration_sec, DENSE_FPS_SECONDS):
        out.append(round(t, 3))
        t += 1.0 / DENSE_FPS
    t = max(t, DENSE_FPS_SECONDS)
    while t < duration_sec:
        out.append(round(t, 3))
        t += SPARSE_INTERVAL
    if len(out) > MAX_FRAMES_PER_VIDEO:
        head = out[:MAX_FRAMES_PER_VIDEO // 2]
        tail = out[MAX_FRAMES_PER_VIDEO // 2:]
        step = max(1, len(tail) // (MAX_FRAMES_PER_VIDEO - len(head)))
        out = head + tail[::step]
        out = out[:MAX_FRAMES_PER_VIDEO]
    return out


def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video {path.name}")
    return cap


def _duration_seconds(cap: cv2.VideoCapture) -> float | None:
    """Compute duration from FPS × frame_count.

    Some containers report one but not the other; some report zeros.
    Returns None in those cases so callers can fall back to the
    poster timestamp.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps > 0 and n_frames > 0:
        return n_frames / fps
    return None


def _decode_at(
    cap: cv2.VideoCapture, seek_target_sec: float
) -> tuple[np.ndarray, float] | None:
    """Seek to ``seek_target_sec`` and decode one frame.

    Returns ``(rgb_uint8_HxWx3, actual_ts_sec)`` or ``None`` if no
    frame could be decoded. cv2 returns BGR; we convert to RGB to
    match the rest of the pipeline.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_target_sec * 1000.0)
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # POS_MSEC after a successful read() reports the *next* frame
    # position. Stepping back one frame would require knowing fps; for
    # the dedup-by-ms set membership we want the position at decode
    # time, so subtract one frame interval if we have fps.
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        pos_ms -= 1000.0 / fps
    return rgb, max(pos_ms / 1000.0, 0.0)


def _downscale(rgb: np.ndarray, max_pixels: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if h * w <= max_pixels:
        return rgb
    scale = (max_pixels / (h * w)) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    pil = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil)


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
    cap = _open(path)
    try:
        duration_sec = _duration_seconds(cap)
        seek_target = ts_sec
        if duration_sec is not None and duration_sec > 0:
            seek_target = min(ts_sec, duration_sec / 2)
        seek_target = max(seek_target, 0.0)

        result = _decode_at(cap, seek_target)
        if result is None:
            # Try the first frame as a fallback — some containers
            # refuse mid-stream seeks (older H.263 .3gp, etc.) but
            # will hand over frame 0 cleanly.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"no decodable frame in {path.name}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb, _ = result
    finally:
        cap.release()

    rgb = _downscale(rgb, max_pixels)
    return rgb, duration_sec


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
    cap = _open(path)
    try:
        duration_sec = _duration_seconds(cap)
        targets = (
            list(timestamps) if timestamps is not None
            else frame_schedule(duration_sec)
        )

        # Track the actual timestamps we've already yielded. With a
        # codec whose keyframe interval > our schedule step (older
        # H.264 / MJPEG / some 2014-era iPhone .mov clips), consecutive
        # seeks land on the same keyframe and we'd otherwise yield the
        # same physical frame twice — wasting the ML work *and*
        # triggering a UNIQUE(file_id, ts_sec) failure when the
        # consumer tries to INSERT both.
        yielded_pts: set[float] = set()

        for ts in targets:
            seek_target = ts
            if duration_sec is not None and duration_sec > 0:
                seek_target = min(seek_target, duration_sec - 0.05)
            seek_target = max(seek_target, 0.0)

            result = _decode_at(cap, seek_target)
            if result is None:
                continue
            rgb, actual = result
            # Round to ms so floating-point noise doesn't defeat the
            # set-membership check.
            key = round(actual, 3)
            if key in yielded_pts:
                continue
            yielded_pts.add(key)

            rgb = _downscale(rgb, max_pixels)
            yield actual, rgb, duration_sec
    finally:
        cap.release()
