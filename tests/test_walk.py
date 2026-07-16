"""Walker exclusions for Apple photo-library bundles."""
from __future__ import annotations

from pathlib import Path

from imagewizard import scan


def _mk(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff\xd8\xff\xe0stub")  # JPEG magic; enough for the walker


def test_photoslibrary_skips_derivatives_keeps_originals(tmp_path):
    lib = tmp_path / "Photos Library.photoslibrary"
    _mk(lib / "originals" / "0" / "real.jpg")
    _mk(lib / "Masters" / "2005" / "master.jpg")   # older-library originals
    _mk(lib / "Previews" / "2014" / "prev.jpg")     # rendered derivative
    _mk(lib / "Thumbnails" / "t.jpg")               # derivative
    _mk(lib / "resources" / "r.jpg")                # derivative

    found = {p.name for p in scan._walk_one_root(lib)}

    assert "real.jpg" in found        # originals/ kept
    assert "master.jpg" in found      # Masters/ kept
    assert "prev.jpg" not in found    # Previews/ skipped (the fix)
    assert "thumb" not in found and "t.jpg" not in found
    assert "r.jpg" not in found


def test_aperture_library_previews_skipped(tmp_path):
    lib = tmp_path / "Aperture Library.aplibrary"
    _mk(lib / "Masters" / "2011" / "orig.jpg")     # Aperture originals
    _mk(lib / "Previews" / "2011" / "prev.jpg")     # derivative
    found = {p.name for p in scan._walk_one_root(lib)}
    assert "orig.jpg" in found
    assert "prev.jpg" not in found


def test_previews_only_skipped_inside_a_library(tmp_path):
    # A user's ordinary folder that happens to be named "Previews" must
    # NOT be skipped — the exclusion only applies inside a photo library.
    root = tmp_path / "my pictures"
    _mk(root / "Previews" / "vacation.jpg")
    found = {p.name for p in scan._walk_one_root(root)}
    assert "vacation.jpg" in found
