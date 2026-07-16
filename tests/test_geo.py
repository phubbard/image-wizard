"""Place + landmark search over the offline gazetteers."""
from __future__ import annotations

import math

import pytest

from imagewizard import geo


def _find(hits, name):
    return next((h for h in hits if h.name.lower() == name.lower()), None)


def test_landmarks_gazetteer_loads():
    lms = geo._load_landmarks()
    # The bundled gazetteer ships with the package; if present it should be
    # substantial. (Empty only if the data file is missing.)
    assert isinstance(lms, list)
    if lms:
        assert len(lms) > 10_000


@pytest.mark.skipif(not geo._load_landmarks(), reason="no landmarks gazetteer bundled")
class TestLandmarkSearch:
    def test_wrigley_field(self):
        # The motivating example: a stadium, not a city.
        hits = geo.search_places("wrigley field", limit=5)
        wf = _find(hits, "Wrigley Field")
        assert wf is not None
        # Chicago-ish coordinates.
        assert abs(wf.lat - 41.95) < 0.2 and abs(wf.lon - (-87.66)) < 0.2

    def test_hyphen_insensitive(self):
        # "notre dame" (space) must match hyphenated Notre-Dame names.
        hits = geo.search_places("notre dame", limit=5)
        assert any("notre" in h.name.lower() for h in hits)

    def test_landmark_carries_kind_in_region(self):
        hits = geo.search_places("wrigley field", limit=3)
        wf = _find(hits, "Wrigley Field")
        assert wf.region == "stadium"


def test_city_search_still_works():
    # Landmarks must not crowd out plain city search.
    hits = geo.search_places("paris", limit=5)
    paris = _find(hits, "Paris")
    assert paris is not None
    assert abs(paris.lat - 48.85) < 0.5


def test_empty_query():
    assert geo.search_places("") == []
    assert geo.search_places("   ") == []
