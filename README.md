# image-wizard

Local, on-device photo indexer for macOS (Apple Silicon). Indexes your photo
library by date, camera, and geolocation, then runs three ML models — all
locally, no cloud calls:

- **YOLO 11** — object detection (80 COCO classes, MPS-accelerated)
- **InsightFace** — face detection + 512-d ArcFace embeddings, clustered with HDBSCAN for iOS-style "People" albums
- **OpenCLIP** — text-to-image search ("dog on a beach")

All metadata, vectors, and thumbnails live in a single SQLite database with
[sqlite-vec](https://github.com/asg017/sqlite-vec) for KNN search.

## Prerequisites

```bash
brew install python@3.12 exiftool uv
```

> **Note:** Use Homebrew's Python 3.12, not the python.org installer — the
> latter lacks SQLite extension support needed by sqlite-vec.

## Install

```bash
git clone <repo-url> && cd image-wizard
uv venv --python python3.12
uv pip install -e ".[dev]"

# Optional: RAW format support (CR2, NEF, ARW, DNG, ...)
uv pip install rawpy
```

## Quick start

```bash
# 1. Create database and directories
image-wizard init

# 2. Scan directories (file discovery + SHA-256 hashing, no ML)
image-wizard scan ~/Photos /Volumes/SD-Card/DCIM

# 3. Run the full ML pipeline
image-wizard index

# 4. Cluster detected faces into people
image-wizard cluster-faces

# 5. Browse your library
image-wizard serve          # http://127.0.0.1:8765
```

All commands are incremental — re-running `scan` + `index` only processes new
or modified files.

## Commands

```
image-wizard init                              Create database + directories
image-wizard scan PATH... [--prune]            Discover files, hash, skip thumbnails
                          [--min-pixels 320]
image-wizard index        [-n LIMIT]           Run ML pipeline on unindexed files
                          [--no-yolo]
                          [--no-faces]
                          [--no-clip]
                          [--workers 4]        Prefetch/decode threads
                          [--prefetch 8]       Queue depth
image-wizard cluster-faces [--min-size 3]      HDBSCAN face clustering
image-wizard search "query" [-k 20]            CLIP text-to-image search
image-wizard stats                             Library counts, cameras, date range
image-wizard serve [--port 8765] [--reload]    Web UI (--reload for development)
image-wizard drop-small [--min-pixels 320]     Remove small images from DB
```

## Web UI

| Page | What it does |
|------|-------------|
| **Timeline** (`/`) | Photos by date, infinite scroll. URL tracks scroll position for back-button. |
| **Search** (`/search`) | Free-text CLIP search, plus dropdowns for object label, camera, person, country. |
| **Photo detail** (`/photo/{id}`) | Full image with toggleable bounding boxes for objects (teal) and faces (yellow). Click a face box to name it inline — propagates to the whole cluster. |
| **Faces** (`/faces`) | Grid of face clusters with naming. |
| **Map** (`/map`) | Geolocated photos on Leaflet/OSM with popups. |

## Pipeline architecture

```
 ┌─────────────────────────────┐
 │  1. Metadata batch          │  exiftool native batch (200 files/call)
 │     + reverse geocoding     │  offline, ~2 MB cities DB
 └──────────┬──────────────────┘
            ▼
 ┌─────────────────────────────┐
 │  2. Prefetch pool           │  N threads: decode (HEIC/RAW/JPEG) + thumbnail
 │     ThreadPoolExecutor      │  bounded queue for backpressure
 └──────────┬──────────────────┘
            ▼
 ┌─────────────────────────────┐
 │  3. GPU inference           │  YOLO (MPS) → CLIP (MPS) → InsightFace (CPU/ONNX)
 │     main thread             │  models pre-warmed before first image
 └──────────┬──────────────────┘
            ▼
 ┌─────────────────────────────┐
 │  4. DB writes               │  serialized SQLite, idempotent (safe to Ctrl-C + restart)
 └─────────────────────────────┘
```

## Stack

| Concern | Choice |
|---------|--------|
| Language | Python 3.12, uv |
| Storage | SQLite + sqlite-vec (vector KNN in SQL) |
| Metadata | exiftool via pyexiftool |
| Image decode | Pillow, pillow-heif, rawpy (optional) |
| Object detection | Ultralytics YOLO 11n, MPS-accelerated |
| Face detect + embed | InsightFace buffalo_l (ArcFace 512-d) |
| Face clustering | HDBSCAN with stable cluster IDs |
| Text search | OpenCLIP ViT-B/32 (laion2b) |
| Reverse geocoding | reverse_geocoder (offline) |
| CLI | Typer |
| Web UI | FastAPI + Jinja2 + htmx (no JS build step) |

## File layout

```
imagewizard/
  cli.py            CLI entrypoint (typer)
  config.py         XDG paths, model cache
  db.py             SQLite schema, sqlite-vec, migrations
  scan.py           Directory walker, SHA-256 hashing, thumbnail filter
  metadata.py       ExifTool wrapper (thread-safe, batch mode)
  geo.py            Offline reverse geocoding
  decode.py         HEIC / RAW / JPEG → RGB numpy
  thumbs.py         512px JPEG thumbnail cache
  pipeline.py       Concurrent ingestion orchestrator
  cluster.py        HDBSCAN face clustering
  search_cli.py     CLIP text search CLI
  models/
    yolo.py         YOLO 11n (lazy singleton)
    faces.py        InsightFace buffalo_l (lazy singleton)
    clip.py         OpenCLIP ViT-B/32 (lazy singleton)
  web/
    app.py          FastAPI application
    templates/      Jinja2 + htmx templates
    static/         favicon, CSS
```

## Data locations

| What | Where |
|------|-------|
| Database | `~/Library/Application Support/image-wizard/imagewizard.sqlite` |
| Thumbnails | `~/Library/Caches/image-wizard/thumbs/` |
| Model cache | `~/Library/Caches/image-wizard/models/` |

Override with `IMAGEWIZARD_DATA_DIR` and `IMAGEWIZARD_CACHE_DIR` env vars.

## Maintenance

```bash
# Re-scan after adding/removing photos
image-wizard scan ~/Photos --prune

# Re-index only new files
image-wizard index

# Re-cluster after new faces are indexed
image-wizard cluster-faces

# Remove old thumbnails from DB (< 320px)
image-wizard drop-small

# Development mode (auto-reload on code changes)
image-wizard serve --reload
```

## License

MIT
