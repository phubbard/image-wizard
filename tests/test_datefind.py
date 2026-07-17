"""Date inference from file paths (imagewizard.datefind)."""
from __future__ import annotations

import pytest

from imagewizard.datefind import infer_date


@pytest.mark.parametrize("path,expect", [
    # YYYY/MM/DD folder structure (Apple .photoslibrary date tree)
    ("/Volumes/photo/Photos Library.photoslibrary/2004/10/25/DSCN0284.JPG", "2004-10-25"),
    ("/lib/2005/8/4/x.jpg", "2005-08-04"),                 # non-zero-padded
    # spelled-out event folders (the "Photos export" layout)
    ("/x/Shimogyo-Ku, Kyoto - Sugiebisucho, August 4, 2005/IMG_4857.JPG", "2005-08-04"),
    ("/x/December 20, 2010/P1000845.JPG", "2010-12-20"),
    ("/x/March 27, 2004/IMG.jpg", "2004-03-27"),
    ("/x/Jan 3 2007/a.jpg", "2007-01-03"),
    # filename-embedded dates
    ("/x/IMG_20040825_143000.jpg", "2004-08-25"),
    ("/x/PXL_20211105.jpg", "2021-11-05"),
    ("/x/Screenshot 2019-05-13.png", "2019-05-13"),
    ("/x/2013-07-04 party.jpg", "2013-07-04"),
    # year/month only → first of the month
    ("/photos/2005/08/IMG.jpg", "2005-08-01"),
])
def test_infers(path, expect):
    got = infer_date(path)
    assert got is not None and got[:10] == expect


def test_filename_time_component():
    # HH:MM:SS is preserved when the filename carries it.
    assert infer_date("/x/IMG_20040825_143007.jpg") == "2004-08-25 14:30:07"


def test_date_only_defaults_to_noon():
    assert infer_date("/x/2004/10/25/a.jpg").endswith("12:00:00")


@pytest.mark.parametrize("path", [
    "/x/IMG_1234.JPG",          # camera sequence, not a date
    "/x/IMG_12345678.jpg",      # 1234-56-78 — not a valid calendar date
    "/x/DSC00089.JPG",
    "/x/99999999.jpg",
    "/x/20043299.jpg",          # month 32 invalid
    "/random/vacation/photo.jpg",
    "/x/1889/01/01/old.jpg",    # year before the plausible range
])
def test_rejects_non_dates(path):
    assert infer_date(path) is None


def test_epoch_ms_filename():
    # Dropbox/Android "Camera Uploads" — filename is an epoch-ms timestamp.
    import datetime as _dt
    got = infer_date("/Volumes/photo/Camera/1347491616193.jpg")
    assert got is not None and got.startswith("2012-09")
    expected = _dt.datetime.fromtimestamp(
        1347491616193 / 1000, tz=_dt.timezone.utc).strftime("%Y-%m-%d")
    assert got[:10] == expected


@pytest.mark.parametrize("path", [
    "/x/9999999999999.jpg",   # 13 digits but decodes to ~2286 (out of range)
    "/x/1234567890.jpg",      # 10 digits — deliberately NOT treated as epoch
])
def test_epoch_ms_guardrails(path):
    assert infer_date(path) is None
