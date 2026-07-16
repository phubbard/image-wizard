#!/usr/bin/env python3
"""Build the bundled offline landmarks gazetteer from Wikidata.

Fetches *notable* points of interest — stadiums, museums, national parks,
monuments, castles, airports, universities, etc. — filtered by Wikidata
sitelink count (a good notability proxy, so we keep Wrigley Field and drop
the local high-school field). Writes a compact gzipped TSV that ships in
the package and is loaded offline at query time by ``imagewizard.geo``.

Run occasionally to refresh:

    python tools/build_landmarks.py

Network is needed only for this build step, never at query time.
"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "imagewizard" / "data" / "landmarks.tsv.gz"
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "image-wizard-landmarks/0.1 (personal photo indexer; offline gazetteer build)"
MIN_SITELINKS = 5          # notability floor
PER_TYPE_LIMIT = 15000     # cap the most-notable per type

# (type QID, short kind label, subclass?). Verified against Wikidata.
# subclass=True traverses P279* so specific subtypes are included — needed
# for "tourist attraction" (most famous attractions are subtypes) and for
# "architectural landmark", where the Eiffel Tower et al. live.
TYPES = [
    ("Q483110",  "stadium",             False),
    ("Q33506",   "museum",              False),
    ("Q207694",  "art museum",          False),
    ("Q2087181", "historic house",      False),
    ("Q570116",  "attraction",          True),
    ("Q2319498", "landmark",            False),
    ("Q1440300", "tower",               False),
    ("Q23413",   "castle",              False),
    ("Q16560",   "palace",              False),
    ("Q57821",   "fortification",       False),
    ("Q4989906", "monument",            False),
    ("Q839954",  "archaeological site", False),
    ("Q1248784", "airport",             False),
    ("Q3918",    "university",          False),
    ("Q22698",   "park",                False),
    ("Q46169",   "national park",       False),
    ("Q174782",  "square",              False),
    ("Q40080",   "beach",               False),
    ("Q12518",   "tower",               False),
    ("Q162875",  "mausoleum",           False),  # Taj Mahal
    ("Q12280",   "bridge",              True),   # subclass: suspension etc. (Golden Gate)
    ("Q2977",    "cathedral",           False),
    ("Q16970",   "church",              False),
    ("Q44539",   "temple",              False),
    ("Q32815",   "mosque",              False),
    ("Q34627",   "synagogue",           False),
    ("Q39715",   "lighthouse",          False),
    ("Q8502",    "mountain",            False),
    ("Q34038",   "waterfall",           False),
]


def fetch_type(qid: str, subclass: bool = False) -> list[dict]:
    p31 = "wdt:P31/wdt:P279*" if subclass else "wdt:P31"
    q = f"""
    SELECT ?item ?name ?coord ?sl WHERE {{
      ?item {p31} wd:{qid} ;
            wdt:P625 ?coord ;
            wikibase:sitelinks ?sl ;
            rdfs:label ?name .
      FILTER(?sl >= {MIN_SITELINKS})
      FILTER(lang(?name) = "en")
    }}
    ORDER BY DESC(?sl)
    LIMIT {PER_TYPE_LIMIT}
    """
    url = ENDPOINT + "?format=json&query=" + urllib.parse.quote(q)
    # Shell out to curl: it uses the system cert store, whereas python.org
    # Python's urllib often can't verify TLS on macOS (no certifi bundle).
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "120",
         "-H", f"User-Agent: {UA}",
         "-H", "Accept: application/sparql-results+json", url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"curl rc={proc.returncode}: {proc.stderr[:200]}")
    data = json.loads(proc.stdout)
    out = []
    for b in data["results"]["bindings"]:
        qid_item = b["item"]["value"].rsplit("/", 1)[-1]
        name = b["name"]["value"]
        wkt = b["coord"]["value"]  # "Point(lon lat)"
        try:
            inner = wkt[wkt.index("(") + 1:wkt.index(")")]
            lon_s, lat_s = inner.split()
            lat, lon = float(lat_s), float(lon_s)
        except (ValueError, IndexError):
            continue
        out.append({
            "qid": qid_item, "name": name, "lat": lat, "lon": lon,
            "sitelinks": int(b["sl"]["value"]),
        })
    return out


def main() -> None:
    best: dict[str, dict] = {}   # qid -> best row (dedupe items matching >1 type)
    for qid, kind, subclass in TYPES:
        for attempt in range(3):
            try:
                rows = fetch_type(qid, subclass)
                break
            except Exception as e:
                print(f"  {qid} attempt {attempt+1} failed: {e}", file=sys.stderr)
                time.sleep(5)
        else:
            print(f"  {qid} ({kind}) — giving up", file=sys.stderr)
            continue
        added = 0
        for r in rows:
            prev = best.get(r["qid"])
            # Keep the higher-sitelink instance; tag with this kind.
            if prev is None or r["sitelinks"] > prev["sitelinks"]:
                r["kind"] = kind
                best[r["qid"]] = r
                added += 1
        print(f"  {qid:10} {kind:20} {len(rows):6} rows ({added} new)  total={len(best)}")
        time.sleep(1)   # be polite to WDQS

    rows = sorted(best.values(), key=lambda r: -r["sitelinks"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        f.write("name\tlat\tlon\tkind\tsitelinks\n")
        for r in rows:
            name = r["name"].replace("\t", " ").replace("\n", " ")
            f.write(f"{name}\t{r['lat']:.5f}\t{r['lon']:.5f}\t{r['kind']}\t{r['sitelinks']}\n")
    print(f"\nwrote {len(rows)} landmarks -> {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
