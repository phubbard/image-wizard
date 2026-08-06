"""AVFoundation poster extraction must be import-guarded and fail soft.

Callers (video.extract_poster) rely on a ``None`` return to fall back to
OpenCV, so extract_poster must never raise — not on a missing file, not on
a non-video, not on a platform without the pyobjc frameworks.
"""
from __future__ import annotations

from imagewizard import video_avf


def test_available_returns_bool():
    assert isinstance(video_avf.available(), bool)


def test_missing_file_returns_none(tmp_path):
    assert video_avf.extract_poster(tmp_path / "nope.mov", timeout=5.0) is None


def test_non_video_returns_none(tmp_path):
    p = tmp_path / "fake.mov"
    p.write_bytes(b"not a real video at all")
    assert video_avf.extract_poster(p, timeout=5.0) is None
