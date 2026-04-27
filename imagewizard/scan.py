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

RAW_EXTS = frozenset({
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
    if ext in VIDEO_EXTS or ext in RAW_EXTS:
        return False
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            return w < min_pixels and h < min_pixels
    except Exception:
        return False


def discover(roots: list[Path]) -> Iterator[Path]:
    """Yield supported image paths under *roots*, skipping dot-dirs and
    Apple Photos library internals.

    No per-file open here — we want enumeration to be cheap so that
    incremental scans (where most files are already indexed and
    unchanged) finish quickly. The dimension-based "is this a
    thumbnail?" check is applied in ``scan()`` only for files that are
    actually new or changed; established files don't pay the cost
    again.

    Skipped subtrees inside `.photoslibrary` packages:
    Thumbnails / resources / private / external / database / scopes
    These hold auto-generated previews and metadata; the actual photos
    live in `originals/`, which we DO want.
    """
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file():
            if root.suffix.lower() in SUPPORTED_EXTS:
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden directories.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            # Inside an Apple Photos library, skip auto-generated subtrees
            # but keep `originals/` (the real photos). These directories
            # double-count the library and create face/object duplicates.
            if ".photoslibrary" in dirpath:
                dirnames[:] = [
                    d for d in dirnames
                    if d not in {
                        "Thumbnails", "resources", "private", "external",
                        "database", "scopes",
                    }
                ]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                if Path(fn).suffix.lower() in SUPPORTED_EXTS:
                    yield Path(dirpath) / fn


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
    files = list(discover(roots))

    # Pre-load every known (path → mtime, size, missing, too_small, id)
    # into memory. The hot path is then a Python dict lookup instead of
    # one indexed-but-still-non-trivial SQL query per file. For a 4M-file
    # library this is ~500 MB of Python objects but trades comfortably
    # against tens of millions of avoided SQLite round trips per
    # incremental scan.
    known: dict[str, tuple[float, int, int, int, int]] = {}
    for r in conn.execute(
        "SELECT path, mtime, size, missing, too_small, id FROM files"
    ):
        known[r["path"]] = (
            r["mtime"], r["size"], r["missing"], r["too_small"], r["id"],
        )

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

            cached = known.get(spath)

            # Fast path: file is known and unchanged. No PIL open, no hash,
            # no SQL — just count and continue. This is the dominant case
            # on incremental scans and dictates overall throughput. Both
            # "unchanged real photo" and "still too small to bother with"
            # take this path.
            if cached is not None:
                m, sz, missing, too_small, fid = cached
                if m == st.st_mtime and sz == st.st_size:
                    if missing:
                        conn.execute("UPDATE files SET missing=0 WHERE id=?", (fid,))
                    if too_small:
                        stats["skipped_small"] += 1
                    else:
                        stats["unchanged"] += 1
                    continue
                # changed — fall through to re-hash / update path below.
                # Re-apply the size filter; the file's bytes changed, so a
                # previously-fine photo could have been overwritten with a
                # thumbnail (or vice versa).
                if _is_too_small(path, min_pixels):
                    conn.execute(
                        """UPDATE files SET size=?, mtime=?, too_small=1
                           WHERE id=?""",
                        (st.st_size, st.st_mtime, fid),
                    )
                    stats["skipped_small"] += 1
                    continue
                try:
                    chash = content_hash(path)
                except OSError:
                    stats["errors"] += 1
                    continue
                mime = mimetypes.guess_type(spath)[0]
                conn.execute(
                    """UPDATE files SET content_hash=?, size=?, mtime=?, mime=?,
                       indexed_at=?, meta_done=0, yolo_done=0, faces_done=0,
                       clip_done=0, missing=0, too_small=0
                       WHERE id=?""",
                    (chash, st.st_size, st.st_mtime, mime, time.time(), fid),
                )
                stats["changed"] += 1
            else:
                # New file — apply the size filter (skips iPhoto / Synology
                # 200×200 thumbnails) before paying for the hash. Tombstone
                # too-small files so the next scan takes the fast path
                # without paying for another PIL header read.
                if _is_too_small(path, min_pixels):
                    mime = mimetypes.guess_type(spath)[0]
                    conn.execute(
                        """INSERT INTO files
                           (path, content_hash, size, mtime, mime,
                            indexed_at, too_small)
                           VALUES (?, '', ?, ?, ?, ?, 1)""",
                        (spath, st.st_size, st.st_mtime, mime, time.time()),
                    )
                    stats["skipped_small"] += 1
                    continue
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
            # Find content_hashes that appear more than once on live (missing=0)
            # rows. Exclude too_small tombstones (they all carry an empty hash
            # by convention so they'd otherwise group together as "duplicates").
            hashes = conn.execute(
                """SELECT content_hash, COUNT(*) AS cnt
                   FROM files
                   WHERE missing = 0 AND too_small = 0 AND content_hash != ''
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

    @parent.command(name="fix-orientations")
    def cmd_fix_orientations(
        workers: int = typer.Option(4, "--workers", "-w", help="Threads for reading image headers."),
    ) -> None:
        """Reset ML flags for files whose stored dimensions don't match the EXIF-rotated image.

        Files indexed before the EXIF orientation fix have bounding boxes
        (faces, YOLO detections, CLIP vectors) computed from the un-rotated
        image. This command opens each file's header, compares the
        exif-transposed dimensions against what's stored, and resets
        yolo_done/faces_done/clip_done for any mismatches. Then re-run
        `image-wizard index` to recompute ML features with correct orientation.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from PIL import Image, ImageOps

        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console(stderr=True)

        try:
            rows = conn.execute(
                "SELECT id, path, width, height FROM files WHERE missing=0 AND width > 0 AND height > 0"
            ).fetchall()
            console.print(f"[dim]checking {len(rows)} files for orientation mismatch...[/dim]")

            def check_one(row):
                """Return (file_id, needs_reset) by comparing stored vs actual rotated dims."""
                try:
                    p = Path(row["path"])
                    if not p.exists():
                        return (row["id"], False)
                    ext = p.suffix.lower()
                    # RAW formats don't have EXIF orientation issues
                    if ext in RAW_EXTS:
                        return (row["id"], False)
                    if ext in {".heic", ".heif"}:
                        try:
                            import pillow_heif
                            pillow_heif.register_heif_opener()
                        except ImportError:
                            pass
                    with Image.open(p) as img:
                        img = ImageOps.exif_transpose(img)
                        actual_w, actual_h = img.size
                    # If stored dimensions disagree, the ML work used the wrong orientation
                    if actual_w != row["width"] or actual_h != row["height"]:
                        return (row["id"], True)
                    return (row["id"], False)
                except Exception:
                    return (row["id"], False)

            reset_count = 0
            error_count = 0

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                console=console,
            ) as prog:
                task = prog.add_task("checking", total=len(rows))
                ids_to_reset: list[int] = []

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = {pool.submit(check_one, r): r for r in rows}
                    for fut in as_completed(futs):
                        prog.advance(task)
                        fid, needs = fut.result()
                        if needs:
                            ids_to_reset.append(fid)

            if ids_to_reset:
                # Also delete stale detections and faces so they don't pile up
                for fid in ids_to_reset:
                    conn.execute("DELETE FROM detections WHERE file_id=?", (fid,))
                    # Delete face vectors first, then faces
                    face_ids = [r[0] for r in conn.execute(
                        "SELECT id FROM faces WHERE file_id=?", (fid,)
                    ).fetchall()]
                    for face_id in face_ids:
                        conn.execute("DELETE FROM vec_faces WHERE rowid=?", (face_id,))
                    conn.execute("DELETE FROM faces WHERE file_id=?", (fid,))
                    conn.execute("DELETE FROM vec_clip WHERE rowid=?", (fid,))
                    conn.execute(
                        """UPDATE files
                           SET yolo_done=0, faces_done=0, clip_done=0,
                               width=0, height=0
                           WHERE id=?""",
                        (fid,),
                    )
                conn.commit()
                reset_count = len(ids_to_reset)

            Console().print(
                f"  reset: {reset_count} files with orientation mismatch"
            )
            if reset_count:
                Console().print(
                    "[dim]run `image-wizard index` to recompute ML features[/dim]"
                )
        finally:
            conn.close()

    @parent.command(name="regen-thumbs")
    def cmd_regen_thumbs(
        workers: int = typer.Option(4, "--workers", "-w", help="Decode threads."),
        force: bool = typer.Option(
            False, "--force",
            help="Re-generate even if a cached thumbnail already exists.",
        ),
        rotated: bool = typer.Option(
            False, "--rotated",
            help="Only re-generate thumbnails whose source EXIF has a "
                 "non-trivial Orientation tag (i.e. needs rotation). Use "
                 "this to repair thumbs cached before orientation handling "
                 "was added to load_image.",
        ),
        camera: str = typer.Option(
            "", "--camera",
            help="Only re-generate thumbs for photos taken with this "
                 "camera_model (exact match).",
        ),
    ) -> None:
        """Regenerate thumbnails.

        Default (no flags): generate any thumbnails that are missing.
        With ``--force`` or ``--rotated`` it will overwrite existing
        cached thumbs, which is how to repair thumbs from older code
        that didn't apply EXIF orientation.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .decode import load_image
        from .thumbs import ensure_thumbnail, thumb_path
        from PIL import Image, ExifTags

        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console(stderr=True)
        try:
            sql = "SELECT f.id, f.path, f.content_hash FROM files f"
            params: list = []
            where = ["f.missing=0"]
            if camera:
                sql += " JOIN photo_meta pm ON pm.file_id = f.id"
                where.append("pm.camera_model = ?")
                params.append(camera)
            sql += " WHERE " + " AND ".join(where)
            rows = conn.execute(sql, params).fetchall()
            console.print(f"{len(rows)} candidate files")

            ORIENT_TAG = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"),
                274,  # well-known tag id; fallback if Pillow lookup fails
            )

            def needs_rotation(path: str) -> bool:
                """True if EXIF Orientation indicates the file needs rotation
                to display correctly. Cheap — only reads the EXIF dict."""
                try:
                    with Image.open(path) as im:
                        exif = im.getexif()
                        return int(exif.get(ORIENT_TAG, 1)) != 1
                except Exception:
                    return False

            # Filter the work list down to what actually needs doing.
            todo = []
            if rotated:
                console.print("[dim]scanning EXIF orientation...[/dim]")
                for r in rows:
                    if needs_rotation(r["path"]):
                        todo.append(r)
                console.print(f"  {len(todo)} files need rotation")
            else:
                for r in rows:
                    if force or not thumb_path(cfg.cache_dir, r["content_hash"]).exists():
                        todo.append(r)

            if not todo:
                console.print("nothing to do")
                return
            console.print(f"regenerating {len(todo)} thumbnails (force={force or rotated})")

            done = 0
            errors = 0

            def regen(row):
                img = load_image(Path(row["path"]))
                # rotated implies force — we know the cached thumb is wrong
                ensure_thumbnail(
                    img, cfg.cache_dir, row["content_hash"],
                    force=force or rotated,
                )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                from rich.progress import Progress, BarColumn, MofNCompleteColumn, SpinnerColumn
                with Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(), console=console) as prog:
                    task = prog.add_task("thumbnails", total=len(todo))
                    futures = {pool.submit(regen, r): r for r in todo}
                    for fut in as_completed(futures):
                        try:
                            fut.result()
                            done += 1
                        except Exception:
                            errors += 1
                        prog.advance(task)
            Console().print(f"  done: {done}, errors: {errors}")
        finally:
            conn.close()

    @parent.command(name="last-crash")
    def cmd_last_crash(
        tail: int = typer.Option(40, "--tail", "-n", help="Lines to show."),
    ) -> None:
        """Show the tail of the index checkpoint log.

        Use this after a silent crash (segfault, OOM kill, supervisor
        kill) to see exactly which file was being processed at the
        moment of death and the recent memory trajectory. The log lives
        at ``<cache_dir>/logs/index.log`` — append-only, fsync'd after
        every write so the last entry survives a hard kill.

        Lines are space-separated:
          <unix_ts> start  <file_id> <path>
          <unix_ts> done   <file_id>
          <unix_ts> error  <file_id> <message>
          <unix_ts> mem    processed=N rss_mb=M
        """
        cfg = config.load()
        log_path = cfg.cache_dir / "logs" / "index.log"
        console = Console()
        if not log_path.exists():
            console.print(f"[yellow]no log at {log_path}[/yellow]")
            console.print("Run `image-wizard index` first.")
            return
        console.print(f"[dim]{log_path}[/dim]")
        # Read the tail without slurping the whole file
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, 64 * 1024)
            f.seek(size - block)
            chunk = f.read().decode("utf-8", errors="replace")
        lines = chunk.splitlines()[-tail:]
        for ln in lines:
            if " error " in ln:
                console.print(f"[red]{ln}[/red]")
            elif " mem " in ln:
                console.print(f"[cyan]{ln}[/cyan]")
            elif ln.startswith(("---",)) or " ---" in ln:
                console.print(f"[bold yellow]{ln}[/bold yellow]")
            else:
                console.print(ln)

    @parent.command(name="list-failures")
    def cmd_list_failures(
        limit: int = typer.Option(50, "--limit", "-n", help="Max rows to show."),
    ) -> None:
        """List files marked as `decode_failed` by the pipeline.

        These are skipped on subsequent `index` runs so we don't waste
        work retrying corrupt or unsupported files. Use `clear-failures`
        to retry them after fixing or replacing the source file.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM files WHERE decode_failed=1"
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT id, path, decode_error FROM files
                   WHERE decode_failed=1
                   ORDER BY id LIMIT ?""",
                (limit,),
            ).fetchall()
            console.print(f"[bold]{total} file(s) marked as decode-failed[/bold]")
            for r in rows:
                err = (r["decode_error"] or "")[:80]
                console.print(f"  #{r['id']:7d}  {r['path']}")
                console.print(f"           [red]{err}[/red]")
            if total > limit:
                console.print(f"... and {total - limit} more (raise --limit)")
        finally:
            conn.close()

    @parent.command(name="clear-failures")
    def cmd_clear_failures(
        path_glob: str = typer.Option(
            "", "--path",
            help="Only clear failures whose path matches this LIKE pattern "
                 "(e.g. '%/iPhone/%'). Default: clear all.",
        ),
    ) -> None:
        """Reset the `decode_failed` flag so files are retried on next index."""
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            if path_glob:
                cur = conn.execute(
                    "UPDATE files SET decode_failed=0, decode_error=NULL "
                    "WHERE decode_failed=1 AND path LIKE ?",
                    (path_glob,),
                )
            else:
                cur = conn.execute(
                    "UPDATE files SET decode_failed=0, decode_error=NULL "
                    "WHERE decode_failed=1"
                )
            console.print(f"cleared {cur.rowcount} failure(s)")
        finally:
            conn.close()

    @parent.command(name="diagnose")
    def cmd_diagnose(
        target: str = typer.Argument(
            ...,
            help="A file id (integer), content hash, or path. Any of the "
                 "three uniquely identifies one indexed photo.",
        ),
    ) -> None:
        """Print everything known about one photo.

        Useful for answering "why does this photo have no faces / no
        objects / no metadata?" — you get pipeline flags, row counts
        across every table touching this file, and whether the source
        file is still on disk.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            # Resolve target → files row.
            row = None
            if target.isdigit():
                row = conn.execute(
                    "SELECT * FROM files WHERE id=?", (int(target),)
                ).fetchone()
            if row is None and len(target) == 64:
                row = conn.execute(
                    "SELECT * FROM files WHERE content_hash=?", (target,)
                ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM files WHERE path=?", (target,)
                ).fetchone()
            if row is None:
                # Fallback: substring match on path
                row = conn.execute(
                    "SELECT * FROM files WHERE path LIKE ? LIMIT 1",
                    (f"%{target}%",),
                ).fetchone()
            if row is None:
                console.print(f"[red]no file matches {target!r}[/red]")
                raise typer.Exit(1)

            fid = row["id"]
            console.print(f"[bold]file #{fid}[/bold]  {row['path']}")
            console.print(f"  content_hash: {row['content_hash']}")
            console.print(f"  size: {row['size']} bytes  mtime: {row['mtime']}")
            console.print(f"  dimensions: {row['width']}x{row['height']}")
            console.print(
                f"  mime: {row['mime']}  missing: {bool(row['missing'])}  "
                f"on disk: {Path(row['path']).exists()}"
            )

            console.print("\n[bold]Pipeline stage flags[/bold]")
            for stage in ("meta_done", "yolo_done", "faces_done", "clip_done"):
                mark = "[green]✓[/green]" if row[stage] else "[red]✗[/red]"
                console.print(f"  {mark} {stage}")
            try:
                if row["decode_failed"]:
                    console.print(
                        f"  [red]✗ decode_failed[/red] — {row['decode_error']}"
                    )
            except (IndexError, KeyError):
                pass  # pre-migration DB

            meta = conn.execute(
                "SELECT * FROM photo_meta WHERE file_id=?", (fid,)
            ).fetchone()
            if meta:
                console.print("\n[bold]photo_meta[/bold]")
                for k in meta.keys():
                    v = meta[k]
                    if v is not None:
                        console.print(f"  {k}: {v}")
            else:
                console.print(
                    "\n[yellow]no photo_meta row[/yellow] "
                    "(metadata stage hasn't produced output)"
                )

            dets = conn.execute(
                "SELECT label, conf FROM detections WHERE file_id=? ORDER BY conf DESC",
                (fid,),
            ).fetchall()
            console.print(f"\n[bold]detections:[/bold] {len(dets)} row(s)")
            for d in dets[:10]:
                console.print(f"  {d['label']:20s} {d['conf']*100:.1f}%")

            faces = conn.execute(
                """SELECT id, cluster_id, person_name, det_score
                   FROM faces WHERE file_id=?""",
                (fid,),
            ).fetchall()
            console.print(f"\n[bold]faces:[/bold] {len(faces)} row(s)")
            for f in faces:
                console.print(
                    f"  face#{f['id']} cluster={f['cluster_id']} "
                    f"name={f['person_name']!r} score={f['det_score']*100:.1f}%"
                )

            clip = conn.execute(
                "SELECT rowid FROM vec_clip WHERE rowid=?", (fid,)
            ).fetchone()
            console.print(
                f"\n[bold]CLIP embedding:[/bold] "
                f"{'present' if clip else 'missing'}"
            )

            # Suggest next action
            missing_stages = [
                s for s in ("meta_done", "yolo_done", "faces_done", "clip_done")
                if not row[s]
            ]
            if missing_stages:
                console.print(
                    f"\n[yellow]suggested:[/yellow] re-run "
                    f"`image-wizard index` — stages {missing_stages} "
                    "haven't completed on this file."
                )
            elif len(dets) == 0 and len(faces) == 0:
                console.print(
                    "\n[green]all stages ran.[/green] "
                    "Zero detections + zero faces means the models genuinely "
                    "found nothing in this image — not a pipeline bug."
                )
        finally:
            conn.close()
