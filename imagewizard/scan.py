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

# Videos are recognized so we can skip them cleanly, but they are NOT indexed:
# the ML pipeline can't decode them, so letting them through just produces
# "cannot identify image file" warnings for every .MOV sibling of a JPG.
SUPPORTED_EXTS = IMAGE_EXTS

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
    dedupe: bool = True,
) -> dict[str, int]:
    """Walk *roots*, insert/update `files`, return summary counts.

    When *dedupe* is True (the default), a newly-discovered file whose
    sha256 matches an already-indexed live row is skipped entirely — no
    new row, no ML rework. If the matching row is tombstoned (missing=1),
    the existing row is re-pointed at the new path and un-tombstoned, so a
    moved file keeps all its prior ML work.
    """
    import sqlite3

    stats = {
        "new": 0, "changed": 0, "unchanged": 0, "missing": 0,
        "errors": 0, "skipped_small": 0,
        "dedup_skipped": 0, "moved": 0,
    }

    # Record the scan roots so the About page can show what's been indexed.
    now = time.time()
    for root in roots:
        rp = str(root.expanduser().resolve())
        conn.execute(
            "INSERT OR REPLACE INTO scan_roots (path, last_scanned_at) VALUES (?, ?)",
            (rp, now),
        )

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
                # new file (no row by path)
                try:
                    chash = content_hash(path)
                except OSError:
                    stats["errors"] += 1
                    continue

                # Hash dedupe: if these bytes are already indexed elsewhere,
                # avoid redoing ML. Prefer a live row (true duplicate) before
                # considering a tombstoned row (moved file).
                if dedupe:
                    live = conn.execute(
                        """SELECT id, path FROM files
                           WHERE content_hash = ? AND missing = 0
                           LIMIT 1""",
                        (chash,),
                    ).fetchone()
                    if live is not None:
                        # Already indexed at another path — skip outright.
                        # Count its path as "seen" so --prune doesn't mark
                        # the canonical copy as missing.
                        paths_seen.add(live["path"])
                        stats["dedup_skipped"] += 1
                        continue

                    dead = conn.execute(
                        """SELECT id FROM files
                           WHERE content_hash = ? AND missing = 1
                           LIMIT 1""",
                        (chash,),
                    ).fetchone()
                    if dead is not None:
                        # File moved: reuse the existing row so prior ML
                        # work (detections, faces, CLIP vector) is kept.
                        mime = mimetypes.guess_type(spath)[0]
                        conn.execute(
                            """UPDATE files
                               SET path=?, size=?, mtime=?, mime=?,
                                   missing=0, indexed_at=?
                               WHERE id=?""",
                            (spath, st.st_size, st.st_mtime, mime,
                             time.time(), dead["id"]),
                        )
                        stats["moved"] += 1
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
        dedupe: bool = typer.Option(
            True, "--dedupe/--no-dedupe",
            help="Skip byte-identical duplicates of already-indexed files. "
                 "Moved files (same hash, old row tombstoned) are re-pointed "
                 "at the new path so prior ML work is preserved.",
        ),
    ) -> None:
        """Scan directories for images/videos and populate the file index."""
        cfg = config.load()
        db.init(cfg.db_path)
        conn = db.connect(cfg.db_path)
        try:
            result = scan(paths, conn, prune=prune,
                          min_pixels=min_pixels, dedupe=dedupe)
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

    @parent.command(name="find-duplicates")
    def cmd_find_duplicates(
        delete: bool = typer.Option(
            False, "--delete",
            help="Actually delete redundant files. Default is dry-run.",
        ),
        keep: str = typer.Option(
            "shortest-path", "--keep",
            help="Which copy to keep per duplicate group: "
                 "shortest-path | longest-path | oldest | newest.",
        ),
        limit: int = typer.Option(
            0, "--limit",
            help="Stop after processing this many duplicate groups (0 = all).",
        ),
    ) -> None:
        """Find files with identical SHA-256 content hash and (optionally) delete redundant copies.

        Exact duplicates are grouped by content_hash. In each group one file
        is chosen as the keeper (by --keep rule); the rest are reported.
        With --delete, the redundant files are unlinked from disk and their
        rows removed from the DB. Missing/tombstoned rows are ignored.
        """
        if keep not in ("shortest-path", "longest-path", "oldest", "newest"):
            raise typer.BadParameter(
                "--keep must be one of: shortest-path, longest-path, oldest, newest"
            )

        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            # Find content_hashes that appear more than once on live (missing=0) rows.
            hashes = conn.execute(
                """SELECT content_hash, COUNT(*) AS cnt
                   FROM files
                   WHERE missing = 0
                   GROUP BY content_hash
                   HAVING cnt > 1
                   ORDER BY cnt DESC"""
            ).fetchall()

            if not hashes:
                console.print("[green]no duplicates found[/green]")
                return

            total_groups = len(hashes)
            total_dup_files = sum(r["cnt"] - 1 for r in hashes)
            console.print(
                f"[yellow]{total_groups} duplicate groups[/yellow] "
                f"covering {total_dup_files} redundant files"
            )
            if not delete:
                console.print("[dim]dry run — pass --delete to actually remove[/dim]")

            def _picker(rows):
                if keep == "shortest-path":
                    return min(rows, key=lambda r: (len(r["path"]), r["path"]))
                if keep == "longest-path":
                    return max(rows, key=lambda r: (len(r["path"]), r["path"]))
                if keep == "oldest":
                    return min(rows, key=lambda r: r["mtime"])
                return max(rows, key=lambda r: r["mtime"])  # newest

            removed = 0
            freed_bytes = 0
            errors = 0

            for i, h in enumerate(hashes):
                if limit and i >= limit:
                    break
                rows = conn.execute(
                    """SELECT id, path, size, mtime
                       FROM files
                       WHERE content_hash = ? AND missing = 0""",
                    (h["content_hash"],),
                ).fetchall()
                if len(rows) < 2:
                    continue

                keeper = _picker(rows)
                losers = [r for r in rows if r["id"] != keeper["id"]]

                console.print(
                    f"\n[bold]{h['content_hash'][:12]}[/bold] "
                    f"({len(rows)} copies)"
                )
                console.print(f"  [green]keep[/green] {keeper['path']}")
                for r in losers:
                    console.print(f"  [red]dup [/red] {r['path']}")

                if delete:
                    for r in losers:
                        try:
                            p = Path(r["path"])
                            if p.exists():
                                p.unlink()
                            # ON DELETE CASCADE cleans detections/faces/vec rows
                            conn.execute("DELETE FROM files WHERE id=?", (r["id"],))
                            removed += 1
                            freed_bytes += r["size"] or 0
                        except OSError as e:
                            errors += 1
                            console.print(f"    [red]error:[/red] {e}")
                    conn.commit()

            if delete:
                mb = freed_bytes / (1024 * 1024)
                console.print(
                    f"\n[green]removed {removed} files[/green] "
                    f"(~{mb:.1f} MB), errors: {errors}"
                )
            else:
                console.print(
                    f"\n[dim]would remove {total_dup_files} files[/dim]"
                )
        finally:
            conn.close()

    @parent.command(name="drop-videos")
    def cmd_drop_videos() -> None:
        """Remove video files (.mov, .mp4, ...) that earlier scans indexed.

        Videos are no longer scanned, but rows from prior runs need a cleanup
        pass. This deletes them outright — ON DELETE CASCADE clears detections,
        faces, and vector rows too.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        try:
            like = " OR ".join(["LOWER(path) LIKE ?"] * len(VIDEO_EXTS))
            params = [f"%{e}" for e in VIDEO_EXTS]
            n = conn.execute(
                f"SELECT COUNT(*) FROM files WHERE {like}", params
            ).fetchone()[0]
            conn.execute(f"DELETE FROM files WHERE {like}", params)
            conn.commit()
            Console().print(f"  dropped: {n} video rows")
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
