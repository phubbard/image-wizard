"""Offline geocoding over the `reverse_geocoder` cities database.

* ``reverse_geocode(lat, lon)`` — nearest city to a coordinate.
* ``search_places(query)`` — forward search: a place name like
  "paris france" → candidate coordinates. City-level (the bundled DB is
  cities with population > 1000; no street addresses), which matches how
  the app tags locations.

Both are fully offline — the ~7 MB cities CSV ships with the library.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)

_rg = None


@dataclass(frozen=True)
class Place:
    city: str
    region: str
    country: str


@dataclass(frozen=True)
class PlaceHit:
    name: str
    region: str
    country: str   # ISO2 code
    lat: float
    lon: float


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


# Common country name / alias → ISO2, so "paris france" and "paris fr"
# both narrow to FR. Not exhaustive — unmatched country tokens simply
# fall through to a plain name search.
_COUNTRY_ALIASES = {
    "usa": "US", "us": "US", "united states": "US", "america": "US",
    "uk": "GB", "united kingdom": "GB", "britain": "GB", "england": "GB",
    "scotland": "GB", "wales": "GB",
    "france": "FR", "germany": "DE", "deutschland": "DE", "italy": "IT",
    "spain": "ES", "espana": "ES", "portugal": "PT", "netherlands": "NL",
    "holland": "NL", "belgium": "BE", "switzerland": "CH", "austria": "AT",
    "ireland": "IE", "greece": "GR", "poland": "PL", "sweden": "SE",
    "norway": "NO", "denmark": "DK", "finland": "FI", "iceland": "IS",
    "canada": "CA", "mexico": "MX", "brazil": "BR", "argentina": "AR",
    "chile": "CL", "peru": "PE", "colombia": "CO",
    "japan": "JP", "china": "CN", "india": "IN", "thailand": "TH",
    "vietnam": "VN", "korea": "KR", "south korea": "KR",
    "australia": "AU", "new zealand": "NZ",
    "russia": "RU", "turkey": "TR", "egypt": "EG", "morocco": "MA",
    "south africa": "ZA", "kenya": "KE", "israel": "IL",
    "czech republic": "CZ", "czechia": "CZ", "hungary": "HU",
    "croatia": "HR", "iceland": "IS", "philippines": "PH",
    "indonesia": "ID", "malaysia": "MY", "singapore": "SG",
}

_cities: list[dict] | None = None
_country_names: dict[str, str] = {}


def _load_cities() -> list[dict]:
    """Load geonamescache's 32k cities (with population) once."""
    global _cities, _country_names
    if _cities is not None:
        return _cities
    try:
        import geonamescache
        gc = geonamescache.GeonamesCache()
        _cities = list(gc.get_cities().values())
        _country_names = {
            cc: info["name"] for cc, info in gc.get_countries().items()
        }
    except Exception as e:
        log.warning("could not load geonamescache for place search: %s", e)
        _cities = []
    return _cities


def country_name(cc: str) -> str:
    """ISO2 code → full country name (falls back to the code)."""
    _load_cities()
    return _country_names.get(cc, cc)


def search_places(query: str, limit: int = 8) -> list[PlaceHit]:
    """Forward-search cities by name, ranked by population.

    Understands a trailing country token: "paris france" narrows to
    cities named ~paris in FR; "paris" alone returns the biggest matches
    worldwide (Paris FR first by population), then the user picks. The
    ``region`` field carries the full country name for the dropdown
    label; precise city/region/country are re-derived by reverse-geocode
    when the location is actually saved.
    """
    query = (query or "").strip().lower()
    if not query:
        return []
    cities = _load_cities()
    if not cities:
        return []

    tokens = query.replace(",", " ").split()
    country_cc: str | None = None
    for take in (2, 1):
        if len(tokens) > take:
            cand = " ".join(tokens[-take:])
            if cand in _COUNTRY_ALIASES:
                country_cc = _COUNTRY_ALIASES[cand]
                tokens = tokens[:-take]
                break
            if take == 1 and len(cand) == 2 and cand.upper() in _country_names:
                country_cc = cand.upper()
                tokens = tokens[:-1]
                break
    name_query = " ".join(tokens).strip()
    if not name_query:
        return []
    first = name_query.split()[0]

    scored: list[tuple[int, int, PlaceHit]] = []
    for c in cities:
        name = c.get("name", "")
        if not name:
            continue
        nl = name.lower()
        if name_query not in nl and not nl.startswith(first):
            continue
        cc = c.get("countrycode", "")
        if country_cc and cc != country_cc:
            continue
        if nl == name_query:
            score = 100
        elif nl.startswith(name_query):
            score = 60
        elif name_query in nl:
            score = 30
        else:
            score = 10
        pop = int(c.get("population", 0) or 0)
        hit = PlaceHit(
            name=name, region=_country_names.get(cc, cc), country=cc,
            lat=float(c["latitude"]), lon=float(c["longitude"]),
        )
        scored.append((score, pop, hit))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [h for _s, _p, h in scored[:limit]]
