# image-wizard

Local, on-device photo indexer for macOS (Apple Silicon). Indexes your photo
library by date, camera, and geolocation, then runs three ML models — all
locally, no cloud calls:

- **YOLO 11** — object detection (80 COCO classes, MPS-accelerated)
- **InsightFace** — face detection + 512-d ArcFace embeddings, clustered with HDBSCAN for iOS-style "People" albums
- **OpenCLIP** — text-to-image search ("dog on a beach")

Videos (`.mov`, `.mp4`, `.m4v`) are indexed too — multiple frames are
sampled (1 fps for the first 60s, then every 10s; capped at 600 per
video) and each gets the full ML treatment. Per-frame detections,
faces, and CLIP embeddings let you ask "where in this video?" and
"who appears in this video?" with timestamps. The browser plays the
file inline via `<video>` for compatible codecs.

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

# 3. Run the full ML pipeline (resumable — Ctrl-C is safe)
image-wizard index

# 4. Cluster detected faces into people
image-wizard cluster-faces

# 5. Browse your library
image-wizard serve          # http://0.0.0.0:8765
```

All commands are incremental:
- `image-wizard rescan` re-walks every previously-scanned directory in one shot
- `image-wizard index` only processes files that haven't completed each stage
- `image-wizard cluster-faces` only processes faces that haven't been clustered

So the daily refresh after dropping new photos onto disk is just:

```bash
image-wizard rescan && image-wizard index && image-wizard cluster-faces
```

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
image-wizard scan ~/A /Volumes/B --walk-workers 16
                                              # parallel walk: each root in its
                                              # own thread. Big win on multi-mount
                                              # SMB / NFS setups where directory
                                              # latency dominates. Defaults to
                                              # min(roots, 8). Overlapping roots
                                              # (e.g. /Volumes/photo and
                                              # /Volumes/photo/Camera) are safe —
                                              # paths are deduped per scan.

# Re-scan every directory you've ever scanned (reads the scan_roots
# table — no need to retype paths). Skips unmounted roots so an
# offline volume doesn't false-positive --prune. Per-root progress
# bars show which mount is the current bottleneck.
image-wizard rescan
image-wizard rescan --no-prune                # don't tombstone missing files
image-wizard rescan --walk-workers 16         # raise concurrency for NAS

# Survey video files across every scanned root: how many on disk vs
# indexed / pending / failed / never-seen. Useful before kicking off a
# big index run on a fresh V2 deploy.
image-wizard list-videos
image-wizard list-videos --list                # also dump every path
image-wizard list-videos --list --state failed # just the tombstoned ones

# Tombstone a single file so the ML pipeline skips it. Use when one
# specific file crashes the indexer and you want to keep going. Accepts
# id, content_hash, full path, or path substring.
image-wizard skip 253751
image-wizard skip 'P1050349_face0.jpg' --reason "InsightFace native crash"

# Bulk purge auto-generated photo-library thumbnails left over from
# pre-exclusion scans (iPhoto Library/Thumbnails, .photoslibrary
# resources, *_face0.jpg per-face crops). Always preview with
# --dry-run first.
image-wizard cleanup-thumbnails --dry-run
image-wizard cleanup-thumbnails

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

Every grid page uses infinite scroll with URL-synced page state, so
clicking a photo and pressing Back returns you to the exact scroll
position you left.

| Page | What it does |
|------|-------------|
| **Timeline** (`/`) | Photos by date with year + multi-month subheaders. |
| **Search** (`/search`) | Free-text CLIP search ("dog on a beach"), plus filter dropdowns for object label, camera, named person, unnamed face cluster, and country. Prev/Next on photo detail walks the search results. |
| **Photo detail** (`/photo/{id}`) | Full image with toggleable bounding boxes for objects (teal) and faces (yellow). Click a face box to name it inline. The Pipeline section in the sidebar shows the four ML stage flags (✓/✗) so you can tell whether missing detections are a pipeline gap or a genuinely empty result. Decode failures are flagged in red with the recorded error. |
| **Person** (`/person/{name}`) | Per-person timeline. Aggregates every face cluster sharing this identity (handles multi-cluster + multi-name same-person cases). Shows a "through the years" face strip and a "names through time" editor for adding name epochs (`Amy Bee` until 2012-07-01, `Amy Smith` after) — caption text on each photo uses the era-appropriate name. |
| **Faces** (`/faces`) | Grid of face clusters with autocomplete naming. Typing an existing name auto-merges clusters. Multi-select for explicit merges. Pagination via infinite scroll. |
| **Cameras** (`/cameras`) → **Camera detail** (`/camera/{model}`) | Camera summaries; per-camera detail page filters to that model. Phones with multiple lens modules get a sub-pill row (Ultra wide / Main / Telephoto / Front) — the same physical lens may appear under several EXIF strings depending on camera mode, and the pills group those together. |
| **Map** (`/map`) | Every geolocated photo on Leaflet/OSM with marker clustering. Click a marker for a popup. |
| **Nearby** (`/nearby?lat=&lon=&radius_km=`) | Photos within a chosen radius of any GPS-tagged photo (click the GPS row on a photo detail to launch a radius prompt). Sorted by distance. |
| **About** (`/about`) | Library stats, scan timestamps, **directory multi-select**: check / uncheck which scanned roots are visible across Timeline / Search / Map / Nearby. The selection persists in the DB. A `⌖ filtered` pill appears in the nav of every page when a strict subset is active. |

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
| Face clustering | HDBSCAN with stable cluster IDs (incremental) |
| Person identity | Custom model — clusters → persons → name epochs over time |
| Text search | OpenCLIP ViT-B/32 (laion2b) |
| Reverse geocoding | reverse_geocoder (offline) |
| Crash diagnostics | psutil (RSS sampling), faulthandler-to-file, append-only checkpoint log |
| CLI | Typer |
| Web UI | FastAPI + Jinja2 + htmx (no JS build step), Leaflet for map |

## File layout

```
imagewizard/
  cli.py            CLI entrypoint (typer); enables faulthandler to a durable file
  config.py         XDG paths, model cache
  db.py             SQLite schema, sqlite-vec, additive migrations
  scan.py           Directory walker, SHA-256 hashing, all maintenance CLIs
  metadata.py       ExifTool wrapper (thread-safe, batch mode)
  geo.py            Offline reverse geocoding
  decode.py         HEIC / RAW / JPEG → RGB numpy (with EXIF orientation)
  thumbs.py         512px JPEG thumbnail cache
  pipeline.py       Concurrent ingestion orchestrator + checkpoint log
  cluster.py        HDBSCAN face clustering (incremental + full modes)
  persons.py        Person identity model + name epochs (date-aware names)
  search_cli.py     CLIP text search CLI
  models/
    yolo.py         YOLO 11n (lazy singleton)
    faces.py        InsightFace buffalo_l (lazy singleton)
    clip.py         OpenCLIP ViT-B/32 (lazy singleton)
  web/
    app.py          FastAPI application
    log_filter.py   Uvicorn log filter (drops unhelpful "Invalid HTTP" warnings)
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
dies without a Python traceback — macOS killing it for memory pressure,
a segfault inside a C library, the terminal closing, etc. Python can't
intercept any of those.

The pipeline therefore writes two diagnostic streams to
`~/Library/Caches/image-wizard/logs/`:

| File | Contents |
|------|----------|
| `index.log` | Per-file lifecycle (`start`, `stage <yolo\|clip\|faces>`, `done`, `error`), RSS snapshot every 250 files, and `throttle` lines when the memory ceiling is hit. fsync'd after every write so the last entry survives a hard kill. |
| `faulthandler.log` | Python frames at SIGSEGV/SIGABRT/SIGTERM/SIGUSR1. Written to a file so Rich's progress redraw can't eat it. Send `kill -USR1 <pid>` to snapshot a hung process without killing it. |

**The diagnostic command:**

```bash
image-wizard last-crash -n 80 --kernel
```

This dumps the tail of `index.log` (with stages colourised), any
`faulthandler.log` traceback, and recent macOS kernel events filtered
to image-wizard / Jetsam (memory pressure) / memorystatus.

**Triage from what you see:**

| Symptom | Likely cause | Next step |
|---------|-------------|-----------|
| Last log line is `start <id>` (no `stage`) | Died during decode — libheif on a malformed HEIC, network mount disconnect, file descriptor exhaustion | `ulimit -n 8192` before `index`, or bisect with `--no-faces --no-clip` |
| Last log line is `stage <id> yolo` or `clip` | MPS/Torch crash | Lower `--workers`, try `PYTORCH_ENABLE_MPS_FALLBACK=1`, or skip with `--no-clip` |
| Last log line is `stage <id> faces` | ONNX / InsightFace | Lower `--workers`, isolate with `--no-faces` |
| `faulthandler.log` has a Python traceback | Native segfault — the traceback's last C-extension frame names the offender | Open that library's stack frame |
| No traceback, `log show` mentions `memorystatus` / `Jetsam` | OS killed for memory pressure | Lower `--max-rss-gb` (default 60 % of RAM) |
| No traceback, no kernel event, no `.ips` file | External signal — terminal close, supervisor, sleep | Run under `caffeinate -i nohup` or in tmux/zellij, check `last` |
| `.ips` file in `~/Library/Logs/DiagnosticReports/` | Genuine native crash | Open the `.ips`; its top frame identifies the failing library |

**Memory throttle:** when RSS exceeds `--max-rss-gb` (default 60 % of
system RAM, can be lowered) the prefetch pool stops submitting new
work and waits for memory to come back down. `throttle` events are
recorded in `index.log` so you can see when this kicks in.

A common safe tuning for very long runs:

```bash
caffeinate -i image-wizard index --workers 4 --max-rss-gb 16
```

## Typical workflows

```bash
# Daily refresh after adding photos to any known location
image-wizard rescan && image-wizard index && image-wizard cluster-faces

# Something looks wrong with one photo — get a full report
image-wizard diagnose <id|path|hash>

# Sideways thumbnails after pulling a newer build
image-wizard regen-thumbs --rotated

# Long, unattended index run on a memory-tight box
caffeinate -i image-wizard index --workers 4 --max-rss-gb 16

# Investigate a silent crash
image-wizard last-crash -n 80 --kernel
```

## Roadmap

Tracked at the top of this list — anything below has been considered
but not yet committed to:

- **Video support, V2 — UI surfacing in progress.** Backend has
  landed: per-frame `frames` table, `frame_id` columns on
  `detections`/`faces`, new `vec_clip_frames` virtual table for
  per-frame CLIP, multi-frame sampling in the pipeline. Still to land:
  film strip on the photo detail page, per-frame timestamps on the
  person timeline ("Alice at 0:23 in beach.mov"), search returning
  in-video moments.
- **Video support, V3+** — scene-cut detection, audio transcription via
  whisper.cpp, speaker identification.
- **Search across name aliases** — currently `/search?person=Amy Smith`
  finds only photos cached with that name, while `/person/Amy Smith`
  aggregates across every name epoch. The search box should do the
  same aggregation.
- **Edit-in-place for name epochs** — currently delete + re-add.
- **`is_nickname` UI** — the data model already supports it; needs a
  checkbox on the add-name form and a styling pass on the person page.

## License

MIT
