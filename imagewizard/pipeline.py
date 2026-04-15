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
"""

from __future__ import annotations

import logging
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


def _vec_bytes(arr: np.ndarray) -> bytes:
    return struct.pack(f"{len(arr)}f", *arr.tolist())


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


def _decode_one(
    file_id: int,
    path: Path,
    content_hash: str,
    flags: dict,
    cache_dir: Path,
) -> PreparedImage:
    """Decode image + thumbnail. Runs in a prefetch thread (no exiftool)."""
    prep = PreparedImage(
        file_id=file_id, path=path, content_hash=content_hash,
        needs_meta=not flags["meta_done"],
        needs_yolo=not flags["yolo_done"],
        needs_faces=not flags["faces_done"],
        needs_clip=not flags["clip_done"],
    )
    try:
        prep.img = load_image(path)
        prep.height, prep.width = prep.img.shape[:2]
        ensure_thumbnail(prep.img, cache_dir, content_hash)
    except Exception as e:
        prep.error = str(e)
    return prep


def _prewarm_models(skip_yolo: bool, skip_faces: bool, skip_clip: bool) -> tuple:
    """Load all models before the pipeline starts so first image isn't slow."""
    console = Console(stderr=True)
    yolo_detect = face_detect = clip_embed = None

    if not skip_yolo:
        console.print("[dim]loading YOLO...[/dim]", end=" ")
        from .models.yolo import detect as yolo_detect
        # trigger weight download / compile by running on a dummy
        yolo_detect(np.zeros((64, 64, 3), dtype=np.uint8), conf_threshold=0.99)
        console.print("[green]ok[/green]")

    if not skip_faces:
        console.print("[dim]loading InsightFace...[/dim]", end=" ")
        from .models.faces import detect_and_embed as face_detect
        face_detect(np.zeros((64, 64, 3), dtype=np.uint8))
        console.print("[green]ok[/green]")

    if not skip_clip:
        console.print("[dim]loading CLIP...[/dim]", end=" ")
        from .models.clip import embed_image as clip_embed
        clip_embed(np.zeros((64, 64, 3), dtype=np.uint8))
        console.print("[green]ok[/green]")

    return yolo_detect, face_detect, clip_embed


def index_files(
    conn: sqlite3.Connection,
    cfg: config.Config,
    limit: int | None = None,
    skip_yolo: bool = False,
    skip_faces: bool = False,
    skip_clip: bool = False,
    workers: int = 4,
    prefetch_depth: int = 8,
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
    query = (
        "SELECT id, path, content_hash, meta_done, yolo_done, faces_done, clip_done "
        f"FROM files WHERE missing=0 AND ({where})"
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
    yolo_detect, face_detect, clip_embed = _prewarm_models(skip_yolo, skip_faces, skip_clip)

    # Re-check which files still need ML work
    rows = conn.execute(query.replace("meta_done=0 OR ", "").replace(" OR meta_done=0", "").replace("meta_done=0", "1=0")).fetchall()
    if not rows:
        # Only metadata was needed
        return {"processed": len(meta_map), "images/sec": 0}

    # --- Phase 3: prefetch decode + GPU inference ---
    stats = {"processed": 0, "errors": 0, "images/sec": 0.0}
    t0 = time.monotonic()

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

        def submit_prefetch():
            for row in rows:
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

        while True:
            fut = prefetch_q.get()
            if fut is None:
                break

            prep: PreparedImage = fut.result()
            prog.update(task, description=f"[cyan]{prep.path.name}")
            prog.advance(task)

            if prep.error or prep.img is None:
                if prep.error:
                    log.warning("decode error %s: %s", prep.path, prep.error)
                stats["errors"] += 1
                continue

            try:
                # Dimensions
                conn.execute(
                    "UPDATE files SET width=?, height=? WHERE id=?",
                    (prep.width, prep.height, prep.file_id),
                )

                # YOLO (MPS)
                if yolo_detect and prep.needs_yolo:
                    # Clear stale detections from any prior partial run
                    conn.execute("DELETE FROM detections WHERE file_id=?", (prep.file_id,))
                    dets = yolo_detect(prep.img)
                    for d in dets:
                        conn.execute(
                            "INSERT INTO detections (file_id, label, conf, x, y, w, h) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (prep.file_id, d.label, d.conf, d.x, d.y, d.w, d.h),
                        )
                    conn.execute("UPDATE files SET yolo_done=1 WHERE id=?", (prep.file_id,))

                # CLIP (MPS)
                if clip_embed and prep.needs_clip:
                    # sqlite-vec virtual tables don't support OR REPLACE
                    conn.execute("DELETE FROM vec_clip WHERE rowid=?", (prep.file_id,))
                    emb = clip_embed(prep.img)
                    conn.execute(
                        "INSERT INTO vec_clip (rowid, embedding) VALUES (?, ?)",
                        (prep.file_id, _vec_bytes(emb)),
                    )
                    conn.execute("UPDATE files SET clip_done=1 WHERE id=?", (prep.file_id,))

                # Faces (CPU/ONNX — uses its own thread pool internally)
                if face_detect and prep.needs_faces:
                    # Clear stale face rows + their vectors from any prior partial run
                    old_face_ids = [
                        r[0] for r in conn.execute(
                            "SELECT id FROM faces WHERE file_id=?", (prep.file_id,)
                        ).fetchall()
                    ]
                    for ofid in old_face_ids:
                        conn.execute("DELETE FROM vec_faces WHERE rowid=?", (ofid,))
                    conn.execute("DELETE FROM faces WHERE file_id=?", (prep.file_id,))

                    raw_faces = face_detect(prep.img)
                    for f in raw_faces:
                        x1, y1, x2, y2 = f.bbox
                        nx = x1 / prep.width
                        ny = y1 / prep.height
                        nw = (x2 - x1) / prep.width
                        nh = (y2 - y1) / prep.height
                        cur = conn.execute(
                            "INSERT INTO faces (file_id, x, y, w, h, det_score) "
                            "VALUES (?, ?, ?, ?, ?, ?)",
                            (prep.file_id, nx, ny, nw, nh, f.det_score),
                        )
                        conn.execute(
                            "INSERT INTO vec_faces (rowid, embedding) VALUES (?, ?)",
                            (cur.lastrowid, _vec_bytes(f.embedding)),
                        )
                    conn.execute("UPDATE files SET faces_done=1 WHERE id=?", (prep.file_id,))

                prep.img = None  # free early
                stats["processed"] += 1

            except Exception as e:
                log.warning("error processing %s: %s", prep.path, e)
                stats["errors"] += 1

        pool.shutdown(wait=True)

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
        workers: int = typer.Option(4, "--workers", "-w", help="Prefetch/decode threads."),
        prefetch: int = typer.Option(8, "--prefetch", help="Queue depth for pre-decoded images."),
    ) -> None:
        """Run the ML pipeline on scanned but unindexed files."""
        cfg = config.load()
        db.init(cfg.db_path)
        conn = db.connect(cfg.db_path)
        try:
            result = index_files(
                conn, cfg,
                limit=limit,
                skip_yolo=skip_yolo,
                skip_faces=skip_faces,
                skip_clip=skip_clip,
                workers=workers,
                prefetch_depth=prefetch,
            )
            db.set_meta(conn, "last_index_at", str(time.time()))
            console = Console()
            for k, v in result.items():
                console.print(f"  {k}: {v}")
        finally:
            conn.close()
