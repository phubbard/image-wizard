"""Ingestion pipeline: runs each ML stage on unprocessed files.

Pipeline stages:

1. **Metadata batch** — exiftool processes all files in one batch call
   (its native batch mode is 10–50x faster than per-file).
2. **Prefetch pool** — N threads decode images + generate thumbnails.
   No exiftool here — it's already done.
3. **GPU inference** — main thread runs YOLO (MPS) → CLIP (MPS) →
   InsightFace (CPU/ONNX, uses its own thread pool) on each pre-decoded
   image. GPU stays fed because images are already waiting in the queue.
4. **DB writes** — serialized on the main thread after each image.

Crash diagnostics:

A persistent checkpoint log is written to ``<cache>/logs/index.log`` —
each line records "start", "done", "error", or "mem" with a timestamp
and the file id/path. After a silent crash (segfault, OOM kill,
process supervisor kill — none of which Python can intercept) tail the
log to see exactly which file was being processed at the moment of
death and what the memory trajectory was. Memory snapshots also fire
every 250 processed files to catch slow leaks early.
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import sqlite3
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from . import config, db
from .decode import load_image
from .geo import reverse_geocode
from .metadata import ExifTool, PhotoMetadata, store_metadata
from .thumbs import ensure_thumbnail

log = logging.getLogger(__name__)

META_BATCH = 200  # exiftool batch size

# Memory-snapshot interval. Cheap (one psutil call) so a tight cadence is
# fine; a leak typically grows by ~1 MB per file when it does, so 250
# files is enough resolution to spot it early.
MEM_SNAPSHOT_EVERY = 250
# How often to flush MPS / ONNX / Python caches so slow third-party
# leaks don't accumulate unbounded over multi-hour runs.
GC_EVERY = 500


def _flush_native_caches() -> None:
    """Best-effort: nudge PyTorch MPS, ONNX, and CPython GC to free cached
    intermediate buffers. Each call is harmless even if the relevant
    library isn't loaded; failures are swallowed."""
    try:
        import torch
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


def _vec_bytes(arr: np.ndarray) -> bytes:
    return struct.pack(f"{len(arr)}f", *arr.tolist())


class CheckpointLog:
    """Append-only crash-diagnostic log.

    Writes one line per file lifecycle event (start/done/error) plus
    periodic memory snapshots. Flushed and fsync'd after every write so
    that a hard kill (OOM, segfault, power loss) leaves the last entry
    on disk — the user can ``tail`` the file after a crash to see the
    exact file that was in flight.

    Format is intentionally trivial (one event per line, space-
    separated) so it's grep- and awk-friendly. Goes under
    ``<cache>/logs/index.log``.
    """

    def __init__(self, cache_dir: Path):
        self.path = cache_dir / "logs" / "index.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)  # line-buffered
        self._lock = threading.Lock()
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process()
        except ImportError:
            pass
        self._write("--- index started", os.getpid())

    def _write(self, *parts) -> None:
        line = f"{time.time():.3f} " + " ".join(str(p) for p in parts) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass  # fsync failure shouldn't kill indexing

    def start(self, file_id: int, path: Path) -> None:
        self._write("start", file_id, path)

    def stage(self, file_id: int, name: str) -> None:
        """Mark the start of a per-file ML stage (yolo, clip, faces).

        After a silent crash the last 'stage' line in the log identifies
        which model was running when the process died — tells you
        whether to suspect MPS/Torch (yolo, clip) vs ONNX (faces) vs
        decode (no stage line yet for this file).
        """
        self._write("stage", file_id, name)

    def done(self, file_id: int) -> None:
        self._write("done", file_id)

    def error(self, file_id: int, msg: str) -> None:
        # Single-line escaping so the log stays grep-friendly.
        msg = msg.replace("\n", " | ")[:200]
        self._write("error", file_id, msg)

    def memory(self, processed: int) -> None:
        if self._proc is None:
            return
        try:
            mi = self._proc.memory_info()
            rss_mb = mi.rss / (1024 * 1024)
            self._write("mem", f"processed={processed}", f"rss_mb={rss_mb:.0f}")
        except Exception:
            pass

    def close(self, msg: str = "shutdown") -> None:
        self._write("---", msg)
        with self._lock:
            self._fh.close()


VIDEO_EXTS = frozenset({".mov", ".mp4", ".avi", ".mkv", ".m4v"})


@dataclass
class PreparedImage:
    file_id: int
    path: Path
    content_hash: str
    img: np.ndarray | None = None
    width: int = 0
    height: int = 0
    meta: PhotoMetadata | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    needs_meta: bool = False
    needs_yolo: bool = False
    needs_faces: bool = False
    needs_clip: bool = False
    error: str | None = None
    # Set when ``path`` is a video — written into files.duration_sec by
    # the consumer thread. None for images.
    duration_sec: float | None = None


def _process_video_frames(
    prep: "PreparedImage",
    conn: sqlite3.Connection,
    yolo_batch,
    clip_batch,
    face_detect,
    batch_size: int = 8,
) -> int:
    """V2 multi-frame sampling for one video.

    The file-level (poster-frame) detections / faces / CLIP have
    already been written by the main consumer loop with ``frame_id IS
    NULL``. This helper layers per-frame data on top: rows in
    ``frames`` plus per-frame detections, faces, and per-frame CLIP
    vectors keyed by ``frame_id``.

    Idempotent: existing frame data for this file is wiped before
    re-population, so a re-index produces the same final state.

    Batches YOLO and CLIP across sampled frames so a long video runs
    them in chunks rather than one frame at a time.
    """
    from .video import iter_frames

    # Wipe any prior frame-level data. Triggers on faces/frames clear
    # their matching vector-table rows automatically, so we only need
    # to drop the parent rows + the FK-less detections row.
    conn.execute(
        """DELETE FROM detections
           WHERE frame_id IN (SELECT id FROM frames WHERE file_id=?)""",
        (prep.file_id,),
    )
    conn.execute(
        """DELETE FROM faces
           WHERE frame_id IN (SELECT id FROM frames WHERE file_id=?)""",
        (prep.file_id,),
    )
    conn.execute("DELETE FROM frames WHERE file_id=?", (prep.file_id,))

    # Buffer (frame_id, rgb) tuples and flush in chunks of batch_size.
    buf: list[tuple[int, np.ndarray]] = []
    n_frames = 0

    def flush():
        if not buf:
            return
        frames_rgb = [rgb for _, rgb in buf]
        if yolo_batch:
            per_image_dets = yolo_batch(frames_rgb)
            for (fid, _rgb), dets in zip(buf, per_image_dets):
                for d in dets:
                    conn.execute(
                        "INSERT INTO detections "
                        "(file_id, frame_id, label, conf, x, y, w, h) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (prep.file_id, fid, d.label, d.conf,
                         d.x, d.y, d.w, d.h),
                    )
        if clip_batch:
            embs = clip_batch(frames_rgb)
            for (fid, _rgb), emb in zip(buf, embs):
                conn.execute(
                    "INSERT INTO vec_clip_frames (rowid, embedding) VALUES (?, ?)",
                    (fid, _vec_bytes(emb)),
                )
        if face_detect:
            for fid, rgb in buf:
                h, w = rgb.shape[:2]
                for f in face_detect(rgb):
                    x1, y1, x2, y2 = f.bbox
                    nx, ny = x1 / w, y1 / h
                    nw, nh = (x2 - x1) / w, (y2 - y1) / h
                    cur = conn.execute(
                        "INSERT INTO faces "
                        "(file_id, frame_id, x, y, w, h, det_score) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (prep.file_id, fid, nx, ny, nw, nh, f.det_score),
                    )
                    conn.execute(
                        "INSERT INTO vec_faces (rowid, embedding) VALUES (?, ?)",
                        (cur.lastrowid, _vec_bytes(f.embedding)),
                    )
        buf.clear()

    for ts, rgb, _duration in iter_frames(prep.path):
        h, w = rgb.shape[:2]
        # OR IGNORE belt-and-braces: iter_frames already dedupes
        # actual PTS values to one decimal-ms, but if a future caller
        # (or a quirky container) produces a (file_id, ts) pair we've
        # already inserted in this run, we silently skip rather than
        # crashing the whole video.
        cur = conn.execute(
            "INSERT OR IGNORE INTO frames (file_id, ts_sec, width, height) "
            "VALUES (?, ?, ?, ?)",
            (prep.file_id, ts, w, h),
        )
        if cur.rowcount == 0:
            continue
        buf.append((cur.lastrowid, rgb))
        n_frames += 1
        if len(buf) >= batch_size:
            flush()
    flush()
    return n_frames


def _decode_one(
    file_id: int,
    path: Path,
    content_hash: str,
    flags: dict,
    cache_dir: Path,
) -> PreparedImage:
    """Decode image (or pluck a video poster) + thumbnail. Runs in a
    prefetch thread (no exiftool).

    For videos we extract a single frame at ~1 second and treat it as
    an image for every downstream stage. The full video file is served
    by the web layer via the existing /full/{id} endpoint.
    """
    prep = PreparedImage(
        file_id=file_id, path=path, content_hash=content_hash,
        needs_meta=not flags["meta_done"],
        needs_yolo=not flags["yolo_done"],
        needs_faces=not flags["faces_done"],
        needs_clip=not flags["clip_done"],
    )
    try:
        if path.suffix.lower() in VIDEO_EXTS:
            from .video import extract_poster
            prep.img, prep.duration_sec = extract_poster(path)
        else:
            prep.img = load_image(path)
        prep.height, prep.width = prep.img.shape[:2]
        ensure_thumbnail(prep.img, cache_dir, content_hash)
    except Exception as e:
        prep.error = str(e)
    return prep


def _prewarm_models(skip_yolo: bool, skip_faces: bool, skip_clip: bool) -> tuple:
    """Load all models before the pipeline starts so first image isn't slow.

    Returns the batched callables; per-image callers wrap a single image
    in a list. Single-image variants on the model modules delegate to
    these so the warm-up path matches the hot path.
    """
    console = Console(stderr=True)
    yolo_batch = face_detect = clip_batch = None
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)

    if not skip_yolo:
        console.print("[dim]loading YOLO...[/dim]", end=" ")
        from .models.yolo import detect_batch as yolo_batch
        yolo_batch([dummy], conf_threshold=0.99)
        console.print("[green]ok[/green]")

    if not skip_faces:
        console.print("[dim]loading InsightFace...[/dim]", end=" ")
        from .models.faces import detect_and_embed as face_detect
        face_detect(dummy)
        console.print("[green]ok[/green]")

    if not skip_clip:
        console.print("[dim]loading CLIP...[/dim]", end=" ")
        from .models.clip import embed_image_batch as clip_batch
        clip_batch([dummy])
        console.print("[green]ok[/green]")

    return yolo_batch, face_detect, clip_batch


def _system_total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 16.0  # safe assumption

def index_files(
    conn: sqlite3.Connection,
    cfg: config.Config,
    limit: int | None = None,
    skip_yolo: bool = False,
    skip_faces: bool = False,
    skip_clip: bool = False,
    workers: int = 4,
    prefetch_depth: int = 8,
    max_rss_gb: float | None = None,
    batch_size: int = 8,
) -> dict[str, int]:
    """Run the ML pipeline with batch metadata, concurrent prefetch, and warm models."""

    # Find files needing work
    where_parts = []
    if not skip_yolo:
        where_parts.append("yolo_done=0")
    if not skip_faces:
        where_parts.append("faces_done=0")
    if not skip_clip:
        where_parts.append("clip_done=0")
    where_parts.append("meta_done=0")

    where = " OR ".join(where_parts)
    # Skip files we've already failed to decode (clearable via
    # `image-wizard clear-failures`) and files tombstoned as too small to
    # be worth indexing (Synology / iPhoto auto-thumbnails).
    query = (
        "SELECT id, path, content_hash, meta_done, yolo_done, faces_done, clip_done "
        f"FROM files WHERE missing=0 AND decode_failed=0 AND too_small=0 "
        f"AND live_photo_of IS NULL AND ({where})"
    )
    if limit:
        query += f" LIMIT {limit}"

    rows = conn.execute(query).fetchall()
    if not rows:
        return {"processed": 0}

    console = Console(stderr=True)

    # --- Phase 1: batch metadata extraction (exiftool native batch) ---
    meta_needed = [r for r in rows if not r["meta_done"]]
    meta_map: dict[int, tuple[PhotoMetadata, str | None, str | None, str | None]] = {}

    if meta_needed:
        console.print(f"[dim]extracting metadata for {len(meta_needed)} files...[/dim]")
        with ExifTool() as et, Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as meta_prog:
            meta_task = meta_prog.add_task("metadata", total=len(meta_needed))
            for i in range(0, len(meta_needed), META_BATCH):
                batch = meta_needed[i:i + META_BATCH]
                paths = [Path(r["path"]) for r in batch]
                metas = et.extract_batch(paths)
                for r, meta in zip(batch, metas):
                    city = region = country = None
                    if meta.lat is not None and meta.lon is not None:
                        place = reverse_geocode(meta.lat, meta.lon)
                        if place:
                            city, region, country = place.city, place.region, place.country
                    meta_map[r["id"]] = (meta, city, region, country)
                    # Write per-file so a Ctrl-C mid-phase doesn't throw away
                    # hours of exiftool work.
                    store_metadata(conn, r["id"], meta, city, region, country)
                conn.commit()
                meta_prog.advance(meta_task, len(batch))
        console.print(f"[green]metadata done[/green] ({len(meta_map)} files)")

    # --- Phase 2: pre-warm ML models ---
    yolo_batch, face_detect, clip_batch = _prewarm_models(skip_yolo, skip_faces, skip_clip)

    # Re-check which files still need ML work
    rows = conn.execute(query.replace("meta_done=0 OR ", "").replace(" OR meta_done=0", "").replace("meta_done=0", "1=0")).fetchall()
    if not rows:
        # Only metadata was needed
        return {"processed": len(meta_map), "images/sec": 0}

    # --- Phase 3: prefetch decode + GPU inference ---
    stats = {"processed": 0, "errors": 0, "images/sec": 0.0}
    t0 = time.monotonic()
    chk = CheckpointLog(cfg.cache_dir)
    chk.memory(0)  # baseline before the loop

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("·"),
        TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("indexing", total=len(rows))

        prefetch_q: queue.Queue[Future | None] = queue.Queue(maxsize=prefetch_depth)
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="decode")

        # Default RSS ceiling: 60% of system RAM. macOS will start sending
        # memory-pressure SIGKILLs (Jetsam) well before we'd OOM in the
        # traditional sense — 60% is a conservative line that leaves room
        # for the OS, browser, etc. without making indexing pointlessly
        # slow on a 64GB+ Mac Studio.
        if max_rss_gb is None:
            max_rss_gb = max(2.0, _system_total_ram_gb() * 0.60)
        rss_ceiling_bytes = int(max_rss_gb * 1024 ** 3)

        try:
            import psutil
            self_proc = psutil.Process()
        except Exception:
            self_proc = None
        # Informational only — this is the ceiling at which the prefetch
        # pool *would* throttle, not actual usage. Typical resident set
        # is a few GB (models + a handful of in-flight decodes).
        console.print(
            f"[dim]throttle ceiling: {max_rss_gb:.1f} GB (informational; "
            f"actual RSS is logged every {MEM_SNAPSHOT_EVERY} files)[/dim]"
        )

        def submit_prefetch():
            for row in rows:
                # Throttle: if our RSS has climbed past the ceiling, wait
                # for it to come back down. Polling is rare here because
                # the queue itself is bounded by prefetch_depth.
                if self_proc is not None:
                    while True:
                        rss = self_proc.memory_info().rss
                        if rss < rss_ceiling_bytes:
                            break
                        chk._write(
                            "throttle",
                            f"rss_mb={rss/1024/1024:.0f}",
                            f"ceiling_mb={rss_ceiling_bytes/1024/1024:.0f}",
                        )
                        time.sleep(0.5)
                fut = pool.submit(
                    _decode_one,
                    row["id"], Path(row["path"]), row["content_hash"],
                    {
                        "meta_done": True,  # already done above
                        "yolo_done": row["yolo_done"] or skip_yolo,
                        "faces_done": row["faces_done"] or skip_faces,
                        "clip_done": row["clip_done"] or skip_clip,
                    },
                    cfg.cache_dir,
                )
                prefetch_q.put(fut)
            prefetch_q.put(None)

        feeder = threading.Thread(target=submit_prefetch, daemon=True)
        feeder.start()

        def flush_batch(buf: list[PreparedImage]) -> None:
            """Run batched YOLO + CLIP across ``buf``, then per-image faces
            and video frames. Each per-image stage runs inside its own
            try/except so one bad file doesn't poison the batch."""
            if not buf:
                return

            # Phase A: write dimensions for every successful decode. The
            # files row should reflect what we observed even if a later
            # ML stage fails.
            for prep in buf:
                conn.execute(
                    "UPDATE files SET width=?, height=?, kind=?, duration_sec=? WHERE id=?",
                    (
                        prep.width, prep.height,
                        "video" if prep.path.suffix.lower() in VIDEO_EXTS else "image",
                        prep.duration_sec,
                        prep.file_id,
                    ),
                )

            # Phase B: batched YOLO over everyone who needs it.
            yolo_items = [p for p in buf if p.needs_yolo and yolo_batch]
            if yolo_items:
                for p in yolo_items:
                    chk.stage(p.file_id, "yolo")
                try:
                    per_image_dets = yolo_batch([p.img for p in yolo_items])
                    for p, dets in zip(yolo_items, per_image_dets):
                        conn.execute(
                            "DELETE FROM detections WHERE file_id=? AND frame_id IS NULL",
                            (p.file_id,),
                        )
                        for d in dets:
                            conn.execute(
                                "INSERT INTO detections (file_id, label, conf, x, y, w, h) "
                                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (p.file_id, d.label, d.conf, d.x, d.y, d.w, d.h),
                            )
                        conn.execute("UPDATE files SET yolo_done=1 WHERE id=?", (p.file_id,))
                except Exception as e:
                    log.warning("yolo batch failed (%d items): %s", len(yolo_items), e)
                    for p in yolo_items:
                        chk.error(p.file_id, f"yolo: {e}")

            # Phase C: batched CLIP.
            clip_items = [p for p in buf if p.needs_clip and clip_batch]
            if clip_items:
                for p in clip_items:
                    chk.stage(p.file_id, "clip")
                try:
                    embs = clip_batch([p.img for p in clip_items])
                    for p, emb in zip(clip_items, embs):
                        # vec_clip uses rowid=files.id and we may be
                        # re-indexing; the files row isn't being
                        # deleted, so triggers don't help. Explicit
                        # DELETE-then-INSERT is the OR REPLACE we'd
                        # use if vec0 supported it.
                        conn.execute("DELETE FROM vec_clip WHERE rowid=?", (p.file_id,))
                        conn.execute(
                            "INSERT INTO vec_clip (rowid, embedding) VALUES (?, ?)",
                            (p.file_id, _vec_bytes(emb)),
                        )
                        conn.execute("UPDATE files SET clip_done=1 WHERE id=?", (p.file_id,))
                except Exception as e:
                    log.warning("clip batch failed (%d items): %s", len(clip_items), e)
                    for p in clip_items:
                        chk.error(p.file_id, f"clip: {e}")

            # Phase D: per-image faces (InsightFace doesn't batch across
            # images cleanly — it bundles its own internal multi-stage
            # detector+aligner+embedder per call).
            if face_detect:
                for p in buf:
                    if not p.needs_faces:
                        continue
                    chk.stage(p.file_id, "faces")
                    try:
                        # DELETE FROM faces fires the trg_faces_del_vec_faces
                        # trigger for each row, atomically clearing vec_faces.
                        conn.execute(
                            "DELETE FROM faces WHERE file_id=? AND frame_id IS NULL",
                            (p.file_id,),
                        )
                        for f in face_detect(p.img):
                            x1, y1, x2, y2 = f.bbox
                            nx = x1 / p.width
                            ny = y1 / p.height
                            nw = (x2 - x1) / p.width
                            nh = (y2 - y1) / p.height
                            cur = conn.execute(
                                "INSERT INTO faces (file_id, x, y, w, h, det_score) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (p.file_id, nx, ny, nw, nh, f.det_score),
                            )
                            conn.execute(
                                "INSERT INTO vec_faces (rowid, embedding) VALUES (?, ?)",
                                (cur.lastrowid, _vec_bytes(f.embedding)),
                            )
                        conn.execute("UPDATE files SET faces_done=1 WHERE id=?", (p.file_id,))
                    except Exception as e:
                        log.warning("faces failed for %s: %s", p.path, e)
                        chk.error(p.file_id, f"faces: {e}")

            # Phase E: video frame sampling (already batched internally).
            for p in buf:
                if p.path.suffix.lower() not in VIDEO_EXTS:
                    continue
                chk.stage(p.file_id, "video-frames")
                try:
                    _process_video_frames(
                        p, conn,
                        yolo_batch, clip_batch, face_detect,
                        batch_size=batch_size,
                    )
                except Exception as e:
                    log.warning(
                        "video frame extraction failed for %s: %s",
                        p.path, e,
                    )
                    chk.error(p.file_id, f"video-frames: {e}")

            # Phase F: lifecycle bookkeeping.
            for p in buf:
                p.img = None
                meta_map.pop(p.file_id, None)
                stats["processed"] += 1
                chk.done(p.file_id)
                if stats["processed"] % MEM_SNAPSHOT_EVERY == 0:
                    chk.memory(stats["processed"])
                if stats["processed"] % GC_EVERY == 0:
                    _flush_native_caches()
                    chk.memory(stats["processed"])

        batch: list[PreparedImage] = []
        while True:
            fut = prefetch_q.get()
            if fut is None:
                flush_batch(batch)
                batch.clear()
                break

            prep: PreparedImage = fut.result()
            prog.update(task, description=f"[cyan]{prep.path.name}")
            prog.advance(task)
            chk.start(prep.file_id, prep.path)

            if prep.error or prep.img is None:
                if prep.error:
                    log.warning("decode error %s: %s", prep.path, prep.error)
                    chk.error(prep.file_id, f"decode: {prep.error}")
                    conn.execute(
                        "UPDATE files SET decode_failed=1, decode_error=? "
                        "WHERE id=?",
                        (prep.error[:500], prep.file_id),
                    )
                stats["errors"] += 1
                continue

            batch.append(prep)
            if len(batch) >= batch_size:
                flush_batch(batch)
                batch.clear()

        pool.shutdown(wait=True)
    chk.close("loop exited cleanly")

    elapsed = time.monotonic() - t0
    if elapsed > 0 and stats["processed"] > 0:
        stats["images/sec"] = round(stats["processed"] / elapsed, 1)

    stats["processed"] += len(meta_map)
    return stats


# ---- CLI registration ----

def register(parent: typer.Typer) -> None:
    @parent.command(name="index")
    def cmd_index(
        limit: int | None = typer.Option(None, "--limit", "-n", help="Max files to process."),
        skip_yolo: bool = typer.Option(False, "--no-yolo", help="Skip object detection."),
        skip_faces: bool = typer.Option(False, "--no-faces", help="Skip face detection."),
        skip_clip: bool = typer.Option(False, "--no-clip", help="Skip CLIP embeddings."),
        workers: int = typer.Option(
            4, "--workers", "-w",
            help="Prefetch/decode threads. Higher uses more RAM. Values "
                 "above 8 risk macOS killing the process for memory "
                 "pressure (Jetsam) — see README 'Debugging silent crashes'.",
        ),
        prefetch: int = typer.Option(8, "--prefetch", help="Queue depth for pre-decoded images."),
        batch_size: int = typer.Option(
            8, "--batch-size", "-b",
            help="GPU batch size for YOLO/CLIP. Higher = better throughput "
                 "on MPS but more memory per flush. 8 is a good default; "
                 "drop to 4 on 16 GB Macs.",
        ),
        max_rss_gb: float | None = typer.Option(
            None, "--max-rss-gb",
            help="Pause the prefetch pool whenever RSS exceeds this many "
                 "GB. Default: 60%% of system RAM. Lower this if macOS "
                 "is killing the process for memory pressure.",
        ),
    ) -> None:
        """Run the ML pipeline on scanned but unindexed files."""
        cfg = config.load()
        db.init(cfg.db_path)
        conn = db.connect(cfg.db_path)
        if workers > 8:
            Console(stderr=True).print(
                f"[yellow]warning:[/yellow] --workers={workers} is high. "
                "On macOS, indices over 8 are commonly SIGKILLed by the "
                "memory-pressure killer. Consider --workers 4-6 if the "
                "process keeps disappearing without an error."
            )
        try:
            result = index_files(
                conn, cfg,
                limit=limit,
                skip_yolo=skip_yolo,
                skip_faces=skip_faces,
                skip_clip=skip_clip,
                workers=workers,
                prefetch_depth=prefetch,
                max_rss_gb=max_rss_gb,
                batch_size=batch_size,
            )
            db.set_meta(conn, "last_index_at", str(time.time()))
            console = Console()
            for k, v in result.items():
                console.print(f"  {k}: {v}")
        finally:
            conn.close()
