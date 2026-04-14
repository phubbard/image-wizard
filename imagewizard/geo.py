"""Offline reverse geocoding: lat/lon → city, region, country.

Uses the `reverse_geocoder` library which bundles a ~2 MB cities database.
The lookup is loaded lazily once and cached for the process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_rg = None


@dataclass(frozen=True)
class Place:
    city: str
    region: str
    country: str


def reverse_geocode(lat: float, lon: float) -> Place | None:
    """Look up the nearest city. Returns None if lat/lon is invalid."""
    global _rg
    if lat is None or lon is None:
        return None
    try:
        if _rg is None:
            import reverse_geocoder as rg
            _rg = rg
        results = _rg.search([(lat, lon)])
        if results:
            r = results[0]
            return Place(city=r["name"], region=r["admin1"], country=r["cc"])
    except Exception as e:
        log.debug("reverse geocode failed for (%s, %s): %s", lat, lon, e)
    return None
