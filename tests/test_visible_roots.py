"""The Timeline's visible-roots filter must self-heal against drift.

Regression: after tidying scan roots + collating the library onto a new
root, the stored `visible_roots` still pointed at the old (now-absent)
export trees, so the `path LIKE 'root/%'` filter excluded the entire
re-pointed library — the Timeline silently collapsed to a handful of
stragglers. The effective filter now intersects with the live scan roots
and falls back to 'show everything' when nothing stored survives.
"""
from __future__ import annotations

from imagewizard.web.app import _effective_visible_roots


def test_all_stored_roots_stale_falls_back_to_show_all():
    # The exact failure mode: stored filter is the old export trees, none
    # of which is a current scan root -> no filter (show everything).
    stored = ["/Volumes/photo", "/Volumes/photo/107_PANA", "/Volumes/photo/Camera"]
    current = {"/Volumes/2TBSSD/photos", "/Volumes/home/Photos/MobileBackup"}
    assert _effective_visible_roots(stored, current) == []


def test_partial_stale_keeps_the_valid_subset():
    stored = ["/Volumes/photo/Camera", "/Volumes/gone"]
    current = {"/Volumes/2TBSSD/photos", "/Volumes/photo/Camera"}
    assert _effective_visible_roots(stored, current) == ["/Volumes/photo/Camera"]


def test_empty_stored_is_no_filter():
    assert _effective_visible_roots([], {"/Volumes/2TBSSD/photos"}) == []


def test_all_valid_roots_are_preserved_in_order():
    stored = ["/a", "/b"]
    assert _effective_visible_roots(stored, {"/a", "/b", "/c"}) == ["/a", "/b"]


def test_no_scan_roots_at_all_shows_everything():
    # Guard against blanking the view if scan_roots is somehow empty.
    assert _effective_visible_roots(["/a"], set()) == []
