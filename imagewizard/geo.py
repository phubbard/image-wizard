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
import gzip
import logging
import os
from dataclasses import dataclass
from pathlib import Path

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


_landmarks: list[dict] | None = None


def _load_landmarks() -> list[dict]:
    """Load the bundled offline landmarks gazetteer once.

    A compact gzipped TSV of notable POIs (stadiums, museums, parks,
    monuments, …) built from Wikidata by ``tools/build_landmarks.py``.
    Ships with the package; loaded lazily. Missing file → empty list
    (landmark search silently disabled, city search still works).
    """
    global _landmarks
    if _landmarks is not None:
        return _landmarks
    _landmarks = []
    p = Path(__file__).parent / "data" / "landmarks.tsv.gz"
    try:
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as f:
                next(f, None)  # header
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        continue
                    name, lat, lon, kind, sl = parts[:5]
                    try:
                        _landmarks.append({
                            "name": name, "name_l": name.lower(),
                            "lat": float(lat), "lon": float(lon),
                            "kind": kind, "sitelinks": int(sl),
                        })
                    except ValueError:
                        continue
    except Exception as e:
        log.warning("could not load landmarks gazetteer: %s", e)
        _landmarks = []
    return _landmarks


def _name_score(nl: str, name_query: str, first: str) -> int | None:
    """Match-quality score for a lower-cased name vs the query, or None
    if it doesn't match at all. Shared by city + landmark search.

    Hyphens are folded to spaces so "notre dame" matches "Notre-Dame de
    Paris" (the query side is normalized the same way in search_places)."""
    nl = nl.replace("-", " ")
    if name_query not in nl and not nl.startswith(first):
        return None
    if nl == name_query:
        return 100
    if nl.startswith(name_query):
        return 60
    if name_query in nl:
        return 30
    return 10


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
    name_query = " ".join(tokens).strip().replace("-", " ")
    if not name_query:
        return []
    first = name_query.split()[0]

    # Both cities and landmarks are matched by name; each carries an
    # importance in [0,1] (city population vs Wikidata sitelinks, each
    # normalized) so the two sources interleave sensibly — a famous
    # landmark ranks alongside a major city, exact name matches first.
    combined: list[tuple[int, float, PlaceHit]] = []

    for c in cities:
        name = c.get("name", "")
        if not name:
            continue
        score = _name_score(name.lower(), name_query, first)
        if score is None:
            continue
        cc = c.get("countrycode", "")
        if country_cc and cc != country_cc:
            continue
        pop = int(c.get("population", 0) or 0)
        hit = PlaceHit(
            name=name, region=_country_names.get(cc, cc), country=cc,
            lat=float(c["latitude"]), lon=float(c["longitude"]),
        )
        combined.append((score, min(1.0, pop / 1_000_000), hit))

    # Landmarks: no country filter (the gazetteer is name+coord+kind). The
    # kind lands in `region` so the dropdown shows "Wrigley Field · stadium"
    # and reads distinctly from a city row. Precise city/region/country get
    # filled by reverse-geocode when the location is actually saved.
    for lm in _load_landmarks():
        score = _name_score(lm["name_l"], name_query, first)
        if score is None:
            continue
        hit = PlaceHit(
            name=lm["name"], region=lm["kind"], country="",
            lat=lm["lat"], lon=lm["lon"],
        )
        combined.append((score, min(1.0, lm["sitelinks"] / 40.0), hit))

    combined.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [h for _s, _imp, h in combined[:limit]]
