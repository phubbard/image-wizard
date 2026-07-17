"""Root reachability probe — the guard that keeps a dropped network mount
from wedging a scan (and holding the refresh lock) forever."""
from __future__ import annotations

import os
import time

from imagewizard import scan


def test_reachable_real_dir(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert scan._root_reachable(tmp_path) is True


def test_reachable_empty_dir(tmp_path):
    # Empty but mounted → still reachable (scandir returns, just no entries).
    assert scan._root_reachable(tmp_path) is True


def test_unreachable_missing_dir(tmp_path):
    assert scan._root_reachable(tmp_path / "does-not-exist") is False


def test_unreachable_on_timeout(tmp_path, monkeypatch):
    # Simulate a hung mount: scandir that blocks past the timeout. The probe
    # must give up and report unreachable rather than blocking the caller.
    real_scandir = os.scandir

    def hanging_scandir(path):
        time.sleep(5.0)
        return real_scandir(path)

    monkeypatch.setattr(scan.os, "scandir", hanging_scandir)
    t0 = time.time()
    result = scan._root_reachable(tmp_path, timeout=0.3)
    elapsed = time.time() - t0
    assert result is False
    assert elapsed < 2.0   # returned promptly, didn't wait on the hang


def test_scan_skips_unreachable_root(tmp_path):
    # A scan over one good root + one unreachable root must finish, index the
    # good root, count the skip, and — even with --prune — NOT tombstone
    # anything for the offline root.
    from PIL import Image
    from imagewizard import db, scan
    good = tmp_path / "good"
    good.mkdir()
    Image.new("RGB", (512, 512), "blue").save(good / "p.jpg")
    bad = tmp_path / "gone"   # never created → unreachable probe

    dbp = tmp_path / "t.sqlite"
    db.init(dbp)
    conn = db.connect(dbp)
    try:
        stats = scan.scan([good, bad], conn, prune=True)
    finally:
        conn.close()
    assert stats["roots_skipped"] == 1
    assert stats["new"] == 1        # good root's image indexed
    assert stats["missing"] == 0    # offline root didn't false-tombstone
