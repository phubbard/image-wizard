"""EXIF metadata extraction using exiftool.

Uses a persistent exiftool subprocess for speed when processing many files.
Falls back gracefully when exiftool is not installed — metadata columns will
just be NULL, and the user gets a warning.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db

log = logging.getLogger(__name__)


@dataclass
class PhotoMetadata:
    taken_at: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    lens: str | None = None
    iso: int | None = None
    aperture: float | None = None
    shutter: str | None = None
    focal_mm: float | None = None
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None


def _str_or_none(val: Any) -> str | None:
    if val is None or val == "":
        return None
    return str(val)


def _float_or_none(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _int_or_none(val: Any) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _parse_exiftool_dict(d: dict[str, Any]) -> PhotoMetadata:
    """Extract what we need from an exiftool JSON dict."""
    # Date: prefer DateTimeOriginal, then CreateDate, then ModifyDate
    taken = None
    for key in ("EXIF:DateTimeOriginal", "EXIF:CreateDate", "EXIF:ModifyDate",
                "QuickTime:CreateDate", "QuickTime:MediaCreateDate"):
        v = d.get(key)
        if v and v != "0000:00:00 00:00:00":
            # exiftool format: "2023:01:15 14:30:00" → ISO
            taken = str(v).replace(":", "-", 2)
            break

    # GPS
    lat = _float_or_none(d.get("Composite:GPSLatitude") or d.get("EXIF:GPSLatitude"))
    lon = _float_or_none(d.get("Composite:GPSLongitude") or d.get("EXIF:GPSLongitude"))

    # Handle GPS ref (S/W = negative)
    lat_ref = d.get("EXIF:GPSLatitudeRef", "N")
    lon_ref = d.get("EXIF:GPSLongitudeRef", "E")
    if lat is not None and str(lat_ref).upper().startswith("S"):
        lat = -abs(lat)
    if lon is not None and str(lon_ref).upper().startswith("W"):
        lon = -abs(lon)

    return PhotoMetadata(
        taken_at=taken,
        camera_make=_str_or_none(d.get("EXIF:Make")),
        camera_model=_str_or_none(d.get("EXIF:Model")),
        lens=_str_or_none(d.get("EXIF:LensModel") or d.get("Composite:LensID")),
        iso=_int_or_none(d.get("EXIF:ISO")),
        aperture=_float_or_none(d.get("EXIF:FNumber")),
        shutter=_str_or_none(d.get("EXIF:ExposureTime")),
        focal_mm=_float_or_none(d.get("EXIF:FocalLength")),
        lat=lat,
        lon=lon,
        alt=_float_or_none(d.get("EXIF:GPSAltitude")),
    )


class ExifTool:
    """Thread-safe wrapper around a persistent exiftool subprocess."""

    def __init__(self) -> None:
        self._et = None
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            import exiftool
            self._et = exiftool.ExifToolHelper()
            self._et.__enter__()
        except FileNotFoundError:
            log.warning("exiftool not found — metadata extraction disabled")
            self._et = None

    def stop(self) -> None:
        if self._et is not None:
            try:
                self._et.__exit__(None, None, None)
            except Exception:
                pass
            self._et = None

    def __enter__(self) -> ExifTool:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def extract(self, path: Path) -> PhotoMetadata:
        if self._et is None:
            return PhotoMetadata()
        with self._lock:
            try:
                results = self._et.get_metadata(str(path))
                if results:
                    return _parse_exiftool_dict(results[0])
            except Exception as e:
                log.debug("exiftool failed for %s: %s", path, e)
        return PhotoMetadata()

    def extract_batch(self, paths: list[Path]) -> list[PhotoMetadata]:
        """Extract metadata for many paths in one exiftool call (fastest)."""
        if self._et is None or not paths:
            return [PhotoMetadata() for _ in paths]
        with self._lock:
            try:
                results = self._et.get_metadata([str(p) for p in paths])
                return [_parse_exiftool_dict(d) for d in results]
            except Exception as e:
                log.debug("exiftool batch failed: %s", e)
        return [PhotoMetadata() for _ in paths]


def store_metadata(
    conn: sqlite3.Connection,
    file_id: int,
    meta: PhotoMetadata,
    city: str | None = None,
    region: str | None = None,
    country: str | None = None,
) -> None:
    """Insert or replace photo_meta for *file_id*."""
    conn.execute(
        """INSERT OR REPLACE INTO photo_meta
           (file_id, taken_at, camera_make, camera_model, lens, iso,
            aperture, shutter, focal_mm, lat, lon, alt, city, region, country)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (file_id, meta.taken_at, meta.camera_make, meta.camera_model,
         meta.lens, meta.iso, meta.aperture, meta.shutter, meta.focal_mm,
         meta.lat, meta.lon, meta.alt, city, region, country),
    )
    conn.execute("UPDATE files SET meta_done=1 WHERE id=?", (file_id,))
