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

Every command is greppable from this README so you can copy-paste straight into
a terminal. Run any command with `--help` for the full option list.

### Setup & ingest

```bash
# Create database + directories
image-wizard init

# Discover files and SHA-256 hash them (no ML work yet)
image-wizard scan ~/Photos /Volumes/SD-Card/DCIM
image-wizard scan ~/Photos --prune            # mark on-disk deletions missing
image-wizard scan ~/Photos --min-pixels 320   # skip tiny thumbnails

# Run the ML pipeline on files that have been scanned but not yet indexed
image-wizard index                            # full pipeline, resumable
image-wizard index -n 1000                    # limit to N files this run
image-wizard index --no-yolo --no-clip        # metadata + faces only
image-wizard index --workers 8 --prefetch 16  # more decode parallelism
```

### Faces & people

```bash
# Cluster new (unclustered) faces into identities. Fast; rerun after each index.
image-wizard cluster-faces
image-wizard cluster-faces --full             # rebuild every cluster from scratch
image-wizard cluster-faces --min-size 3       # tighter clusters (default: 3)
```

### Browse

```bash
# Web UI
image-wizard serve                            # http://0.0.0.0:8765
image-wizard serve --host 127.0.0.1           # restrict to localhost
image-wizard serve --port 9000                # custom port
image-wizard serve --reload                   # auto-reload on code changes

# CLI CLIP search (web UI has more filters)
image-wizard search "dog on a beach"
image-wizard search "sunset" -k 20
```

### Inspect & diagnose

```bash
# Library counts, cameras, date range
image-wizard stats

# Everything known about one photo: stage flags, row counts across all
# tables, CLIP embedding presence, disk status. Accepts id/hash/path.
image-wizard diagnose 267428
image-wizard diagnose /Users/pfh/Photos/IMG_0036.JPG
image-wizard diagnose c873bad96d2e3eab...
```

### Maintenance & repair

```bash
# Regenerate thumbnails
image-wizard regen-thumbs                     # missing thumbs only
image-wizard regen-thumbs --force             # overwrite all (slow)
image-wizard regen-thumbs --rotated           # only thumbs with non-trivial
                                              # EXIF orientation (repairs sideways
                                              # thumbs cached by older code)
image-wizard regen-thumbs --camera "iPhone"   # scope to one camera

# Clean up the index
image-wizard drop-small --min-pixels 320      # remove small images from DB
image-wizard drop-videos                      # remove .mov/.mp4 rows
image-wizard fix-orientations                 # reset ML flags for files whose
                                              # stored dims don't match rotated image
image-wizard find-duplicates                  # list files sharing a content_hash
image-wizard find-duplicates --delete         # also remove redundant copies

# Files that failed to decode are tombstoned so subsequent index runs skip
# them. Inspect / retry:
image-wizard list-failures                    # paths + recorded errors
image-wizard clear-failures                   # retry every failed file next index
image-wizard clear-failures --path '%/iPhone/%'  # retry just one subtree
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

## Debugging silent crashes

`image-wizard index` is long-running and touches several native
extensions (PyTorch/MPS, ONNX runtime, Pillow, libheif). Sometimes it
dies without a Python traceback — the usual culprits are macOS killing
the process under memory pressure, or a segfault inside a C library.
Python can't intercept either case.

To make these debuggable, the pipeline writes an append-only checkpoint
log to ``~/Library/Caches/image-wizard/logs/index.log``. Each file's
lifecycle gets a line (``start`` / ``done`` / ``error``) and an RSS
snapshot fires every 250 processed files. Each line is fsync'd before
moving on so the last entry survives a hard kill.

After a crash:

```bash
# What was the last file in flight + recent memory trajectory?
image-wizard last-crash             # tail 40 lines
image-wizard last-crash -n 200      # more

# OS-level crash dumps (macOS): segfaults land here as .ips files
ls -lat ~/Library/Logs/DiagnosticReports/ | head
```

If memory is climbing in the ``mem`` snapshots, lower the worker count
(``--workers 4``) and/or skip CLIP for the first pass
(``--no-clip``) — CLIP keeps the largest tensors resident.

**Diagnostic workflow after a silent death:**

```bash
# Per-file lifecycle, native traceback, and recent kernel events
image-wizard last-crash -n 80 --kernel
```

The output combines three sources: the `index.log` checkpoint (last
file + stage in flight, recent RSS), the `faulthandler.log` (Python
frames at SIGSEGV/SIGABRT/SIGTERM — survives Rich's progress redraw
because it's written to a dedicated file), and `log show --last 1h`
filtered to image-wizard / Jetsam events.

**Distinguishing crash modes:**

| Symptom | Likely cause | Next step |
|---------|-------------|-----------|
| Last log line is `start <id>` (no `stage`) | Died during decode (libheif on a malformed HEIC, network mount disconnect on `/Volumes/...`, file descriptor exhaustion) | Try `ulimit -n 8192` before `index`, or `--no-faces --no-clip` to bisect |
| Last log line is `stage <id> {yolo,clip}` | MPS/Torch crash — flaky on long runs | Lower `--workers`, run `--no-clip` separately, or set `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Last log line is `stage <id> faces` | ONNX runtime / InsightFace | Same: lower workers, isolate with `--no-faces` |
| `faulthandler.log` has a Python traceback | Native segfault — the traceback points at the C extension that did it | Open the named library's stack frame |
| No traceback, `log show` mentions `memorystatus`/`Jetsam` | OS killed for memory pressure | Lower `--max-rss-gb` |
| No traceback, no kernel event, no .ips file | External signal (terminal close, supervisor, sleep) | Run under `caffeinate -i` and `nohup`/`tmux`, check `last` output |
| `.ips` file in `~/Library/Logs/DiagnosticReports/` | True native crash | The `.ips` stack trace identifies the offending library |

The pipeline auto-throttles: when RSS climbs past `--max-rss-gb`
(default 60% of system RAM) the prefetch pool stops submitting new
work and waits for memory to come back down. Throttle events are
recorded as `throttle` lines in the index log.

## Typical workflows

```bash
# After adding new photos on disk
image-wizard scan ~/Photos --prune
image-wizard index
image-wizard cluster-faces

# Something looks wrong with one photo — get a full report
image-wizard diagnose <id|path|hash>

# Sideways thumbnails after pulling a newer build
image-wizard regen-thumbs --rotated
```

## License

MIT
