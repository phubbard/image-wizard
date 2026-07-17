"""Infer a capture date from a file's path when EXIF has none.

Old scans and photos from early digital cameras (2002–2006 era) often
carry no EXIF ``DateTimeOriginal``, yet the library filed them into
date-bearing folders (``2004/10/25/``) or event folders ("August 4,
2005"), or the camera embedded the date in the filename
(``IMG_20040825_143000.jpg``). This recovers that date so the photo lands
in the right place in the timeline.

``infer_date(path)`` returns an ISO-ish ``"YYYY-MM-DD HH:MM:SS"`` string
(noon when only a date is known) or ``None``. All candidates are
calendar-validated and range-checked, so a run of digits that isn't a real
date (``IMG_12345678``) is rejected rather than mis-parsed.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone

# Plausible capture years. Digital photos didn't exist before ~1990; the
# upper bound guards against random digit runs. (Scanned-in older prints
# still usually carry a scan-era folder/date, not a 1950s one.)
_MIN_YEAR, _MAX_YEAR = 1990, 2035

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _ok(y: int, m: int, d: int) -> bool:
    if not (_MIN_YEAR <= y <= _MAX_YEAR):
        return False
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def _iso(y: int, m: int, d: int, hh: int = 12, mm: int = 0, ss: int = 0) -> str:
    return f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}:{ss:02d}"


# IMG_20040825_143000 / PXL_20040825 / 20040825_143000 / VID_20040825 …
_FN_DATETIME = re.compile(
    r"(?<!\d)(19\d\d|20\d\d)(\d{2})(\d{2})(?:[_\-]?(\d{2})(\d{2})(\d{2})?)?(?!\d)"
)
# 2004-08-25 or 2004_08_25 (separated), anywhere in the name
_FN_DASHED = re.compile(r"(?<!\d)(19\d\d|20\d\d)[._\-](\d{1,2})[._\-](\d{1,2})(?!\d)")


def _date_from_filename(name: str) -> str | None:
    for m in _FN_DATETIME.finditer(name):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _ok(y, mo, d):
            hh = int(m.group(4)) if m.group(4) else 12
            mm = int(m.group(5)) if m.group(5) else 0
            ss = int(m.group(6)) if m.group(6) else 0
            if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
                hh, mm, ss = 12, 0, 0
            return _iso(y, mo, d, hh, mm, ss)
    for m in _FN_DASHED.finditer(name):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _ok(y, mo, d):
            return _iso(y, mo, d)
    return None


# /2004/10/25/  (zero-padded or not), consecutive path segments
_PATH_YMD = re.compile(r"/(19\d\d|20\d\d)/(\d{1,2})/(\d{1,2})(?=/)")
# /2004/10/ (year+month only)
_PATH_YM = re.compile(r"/(19\d\d|20\d\d)/(\d{1,2})(?=/)")
# "August 4, 2005" / "Aug 4 2005" / "December 20, 2010" in a path segment
_MONTH_DAY_YEAR = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(19\d\d|20\d\d)\b",
    re.IGNORECASE,
)
# "August 2005" (month + year, no day)
_MONTH_YEAR = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(19\d\d|20\d\d)\b", re.IGNORECASE
)


def _date_from_path(path: str) -> str | None:
    # Prefer the most specific: full Y/M/D path segments.
    for m in _PATH_YMD.finditer(path):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _ok(y, mo, d):
            return _iso(y, mo, d)
    # Then a spelled-out "Month D, YYYY" event folder.
    for m in _MONTH_DAY_YEAR.finditer(path):
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            y, d = int(m.group(3)), int(m.group(2))
            if _ok(y, mo, d):
                return _iso(y, mo, d)
    # Then year/month only → first of the month.
    for m in _PATH_YM.finditer(path):
        y, mo = int(m.group(1)), int(m.group(2))
        if _ok(y, mo, 1):
            return _iso(y, mo, 1)
    for m in _MONTH_YEAR.finditer(path):
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            y = int(m.group(2))
            if _ok(y, mo, 1):
                return _iso(y, mo, 1)
    return None


def _date_from_epoch_filename(name: str) -> str | None:
    """A filename that *is* a 13-digit epoch-millisecond timestamp — the
    Dropbox / Android "Camera Uploads" pattern, e.g. ``1347491616193.jpg``.

    Only 13-digit (millisecond) values are accepted, not 10-digit seconds:
    a bare 10-digit filename is too easily an ID that coincidentally lands
    in the plausible-date range. The decoded date is still range-checked.
    """
    m = re.match(r"(\d{13})(?:\D|$)", name)
    if not m:
        return None
    try:
        dt = datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc)
    except (OSError, ValueError, OverflowError):
        return None
    if not (_MIN_YEAR <= dt.year <= _MAX_YEAR):
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def infer_date(path: str) -> str | None:
    """Best-effort capture date for a path, or None.

    Precedence: a date embedded in the *filename* (most specific to the
    file) — an explicit ``YYYYMMDD`` stamp, then a 13-digit epoch-ms name —
    wins; then a ``YYYY/MM/DD`` folder path or spelled-out event folder;
    then year/month only. Returns ``"YYYY-MM-DD HH:MM:SS"``.
    """
    import os
    name = os.path.basename(path)
    return (_date_from_filename(name) or _date_from_epoch_filename(name)
            or _date_from_path(path))
