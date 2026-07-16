"""On-device OCR via Apple's Vision framework.

macOS only — uses ``VNRecognizeTextRequest`` through PyObjC, so text
recognition runs on the Apple Neural Engine / GPU with no cloud calls
and no per-image cost. Recognizes printed and handwritten text in the
photo (street signs, storefronts, book covers, whiteboards,
screenshots) so it can be full-text searched.

Graceful when unavailable: ``available()`` is False off macOS or when
the PyObjC Vision bindings aren't installed, and callers skip OCR.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_checked = False
_ok = False


def available() -> bool:
    """True if Apple Vision OCR can run in this process."""
    global _checked, _ok
    if _checked:
        return _ok
    _checked = True
    try:
        import Vision  # noqa: F401
        import Quartz  # noqa: F401
        _ok = True
    except Exception:
        _ok = False
    return _ok


def recognize_text(image_path: Path, accurate: bool = True) -> str:
    """Return the text recognized in an image file, newline-joined.

    Reads the file directly via CGImageSource (efficient — no full
    Python decode). Returns "" on any failure or if no text is found.
    Pass the 512px thumbnail for speed; it's enough for the sign- and
    label-sized text that matters in a photo library.
    """
    if not available():
        return ""
    try:
        import Vision
        import Quartz
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(str(image_path))
        src = Quartz.CGImageSourceCreateWithURL(url, None)
        if src is None:
            return ""
        cg = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if cg is None:
            return ""
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg, None
        )
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(
            Vision.VNRequestTextRecognitionLevelAccurate
            if accurate else Vision.VNRequestTextRecognitionLevelFast
        )
        req.setUsesLanguageCorrection_(True)
        ok = handler.performRequests_error_([req], None)
        if not ok:
            return ""
        lines: list[str] = []
        results = req.results() or []
        for obs in results:
            cands = obs.topCandidates_(1)
            if cands and len(cands):
                s = cands[0].string()
                if s:
                    lines.append(str(s))
        return "\n".join(lines)
    except Exception as e:
        log.debug("OCR failed for %s: %s", image_path, e)
        return ""
