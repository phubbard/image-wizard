"""Directory scanning: walk, hash, discover new/changed/removed files.

`scan()` is purely filesystem I/O — no ML. It populates the `files` table so
later stages (metadata, YOLO, faces, CLIP) know what to process.

Incremental logic:
* A file is "new" if its path has never been seen.
* A file is "changed" if its (mtime, size) differ from the stored record.
  On change we recompute the content_hash and reset the ML stage flags.
* A file is "missing" if its path was previously indexed but no longer exists
  on disk. We set `missing=1` rather than deleting.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import time
from pathlib import Path
from typing import Iterator

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from . import config, db

IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp",
    ".heic", ".heif", ".avif",
    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef",
})

VIDEO_EXTS = frozenset({".mov", ".mp4", ".avi", ".mkv", ".m4v"})

SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS

CHUNK = 1 << 16  # 64 KB hash chunks


def content_hash(path: Path) -> str:
    """Stream SHA-256 of the whole file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


MIN_PIXELS_DEFAULT = 320  # skip images where BOTH dimensions are below this


def _is_too_small(path: Path, min_pixels: int) -> bool:
    """Fast dimension check — reads only the image header, not full decode."""
    if min_pixels <= 0:
        return False
    ext = path.suffix.lower()
    # Skip dimension check for RAW and video — they're never thumbnails
    if ext in VIDEO_EXTS or ext in {".cr2", ".cr3", ".nef", ".arw", ".dng",
                                     ".orf", ".rw2", ".raf", ".pef"}:
        return False
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            return w < min_pixels and h < min_pixels
    except Exception:
        return False


def discover(roots: list[Path], min_pixels: int = MIN_PIXELS_DEFAULT) -> Iterator[Path]:
    """Yield supported image/video paths under *roots*, skipping dot-dirs
    and images smaller than *min_pixels* in both dimensions."""
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file():
            if root.suffix.lower() in SUPPORTED_EXTS:
                if not _is_too_small(root, min_pixels):
                    yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                if Path(fn).suffix.lower() in SUPPORTED_EXTS:
                    p = Path(dirpath) / fn
                    if not _is_too_small(p, min_pixels):
                        yield p


def scan(
    roots: list[Path],
    conn: "db.sqlite3.Connection",
    prune: bool = False,
    min_pixels: int = MIN_PIXELS_DEFAULT,
) -> dict[str, int]:
    """Walk *roots*, insert/update `files`, return summary counts."""
    import sqlite3

    stats = {"new": 0, "changed": 0, "unchanged": 0, "missing": 0, "errors": 0, "skipped_small": 0}

    paths_seen: set[str] = set()
    files = list(discover(roots, min_pixels=min_pixels))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=Console(stderr=True),
    ) as prog:
        task = prog.add_task("scanning", total=len(files))
        for path in files:
            prog.advance(task)
            spath = str(path)
            paths_seen.add(spath)

            try:
                st = path.stat()
            except OSError:
                stats["errors"] += 1
                continue

            row = conn.execute(
                "SELECT id, mtime, size, content_hash FROM files WHERE path = ?",
                (spath,),
            ).fetchone()

            if row is not None:
                if row["mtime"] == st.st_mtime and row["size"] == st.st_size:
                    # unchanged — make sure it's not tombstoned
                    if conn.execute(
                        "SELECT missing FROM files WHERE id=?", (row["id"],)
                    ).fetchone()["missing"]:
                        conn.execute("UPDATE files SET missing=0 WHERE id=?", (row["id"],))
                    stats["unchanged"] += 1
                    continue
                # changed — recompute hash, reset stage flags
                try:
                    chash = content_hash(path)
                except OSError:
                    stats["errors"] += 1
                    continue
                mime = mimetypes.guess_type(spath)[0]
                conn.execute(
                    """UPDATE files SET content_hash=?, size=?, mtime=?, mime=?,
                       indexed_at=?, meta_done=0, yolo_done=0, faces_done=0,
                       clip_done=0, missing=0
                       WHERE id=?""",
                    (chash, st.st_size, st.st_mtime, mime, time.time(), row["id"]),
                )
                stats["changed"] += 1
            else:
                # new file
                try:
                    chash = content_hash(path)
                except OSError:
                    stats["errors"] += 1
                    continue
                mime = mimetypes.guess_type(spath)[0]
                conn.execute(
                    """INSERT INTO files (path, content_hash, size, mtime, mime, indexed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (spath, chash, st.st_size, st.st_mtime, mime, time.time()),
                )
                stats["new"] += 1

    if prune:
        existing_paths = {
            r[0] for r in conn.execute(
                "SELECT path FROM files WHERE missing=0"
            ).fetchall()
        }
        gone = existing_paths - paths_seen
        for p in gone:
            conn.execute("UPDATE files SET missing=1 WHERE path=?", (p,))
        stats["missing"] = len(gone)

    return stats


# ---- CLI registration ----

def register(parent: typer.Typer) -> None:
    @parent.command(name="scan")
    def cmd_scan(
        paths: list[Path] = typer.Argument(
            ..., help="Directories or files to scan."
        ),
        prune: bool = typer.Option(
            False, "--prune", help="Mark files not found on disk as missing."
        ),
        min_pixels: int = typer.Option(
            MIN_PIXELS_DEFAULT, "--min-pixels",
            help="Skip images where BOTH dimensions are below this (0 to disable).",
        ),
    ) -> None:
        """Scan directories for images/videos and populate the file index."""
        cfg = config.load()
        db.init(cfg.db_path)
        conn = db.connect(cfg.db_path)
        try:
            result = scan(paths, conn, prune=prune, min_pixels=min_pixels)
            console = Console()
            for k, v in result.items():
                console.print(f"  {k}: {v}")
        finally:
            conn.close()

    @parent.command(name="drop-small")
    def cmd_drop_small(
        min_pixels: int = typer.Option(
            MIN_PIXELS_DEFAULT, "--min-pixels",
            help="Remove indexed files where BOTH dimensions are below this.",
        ),
    ) -> None:
        """Remove already-indexed thumbnails/small images from the database."""
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        try:
            rows = conn.execute(
                "SELECT id, path, width, height FROM files WHERE missing=0 AND width>0 AND height>0"
            ).fetchall()
            dropped = 0
            for r in rows:
                if r["width"] < min_pixels and r["height"] < min_pixels:
                    conn.execute("DELETE FROM files WHERE id=?", (r["id"],))
                    dropped += 1
            Console().print(f"  dropped: {dropped} small images (< {min_pixels}px)")
        finally:
            conn.close()

    @parent.command(name="regen-thumbs")
    def cmd_regen_thumbs(
        workers: int = typer.Option(4, "--workers", "-w", help="Decode threads."),
    ) -> None:
        """Regenerate missing thumbnails (e.g. after clearing the cache)."""
        from concurrent.futures import ThreadPoolExecutor
        from .decode import load_image
        from .thumbs import ensure_thumbnail, thumb_path

        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console(stderr=True)
        try:
            rows = conn.execute(
                "SELECT id, path, content_hash FROM files WHERE missing=0"
            ).fetchall()
            # Filter to only those missing a thumbnail
            todo = [
                r for r in rows
                if not thumb_path(cfg.cache_dir, r["content_hash"]).exists()
            ]
            console.print(f"{len(todo)} thumbnails to regenerate")
            if not todo:
                return

            done = 0
            errors = 0

            def regen(row):
                img = load_image(Path(row["path"]))
                ensure_thumbnail(img, cfg.cache_dir, row["content_hash"])

            with ThreadPoolExecutor(max_workers=workers) as pool:
                from rich.progress import Progress, BarColumn, MofNCompleteColumn, SpinnerColumn
                with Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(), console=console) as prog:
                    task = prog.add_task("thumbnails", total=len(todo))
                    futures = {pool.submit(regen, r): r for r in todo}
                    for fut in futures:
                        try:
                            fut.result()
                            done += 1
                        except Exception:
                            errors += 1
                        prog.advance(task)
            Console().print(f"  done: {done}, errors: {errors}")
        finally:
            conn.close()
