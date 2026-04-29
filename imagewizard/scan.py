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
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import typer
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn,
    TimeElapsedColumn,
)

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

# Videos are now supported via PyAV + FFmpeg. The pipeline extracts a
# single poster frame at ~1s and runs the same ML stages on it, so a
# video's row in `files` looks just like a photo's except for kind=video
# and a populated duration_sec.
SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS


def kind_for_ext(ext: str) -> str:
    """Return 'video' for known video extensions, otherwise 'image'.

    Used at scan time to set `files.kind` and at decode time to pick
    between the still-image and video-poster paths.
    """
    return "video" if ext.lower() in VIDEO_EXTS else "image"

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


import re

# Filenames that are *always* auto-generated crop thumbnails — never
# real photos. iPhoto / Photos extract `<name>_face0.jpg` etc. for
# every detected face and litter them throughout the library; running
# our own face detector on them is at best wasted work and at worst
# crashes InsightFace's ONNX runtime (we lost a 7000-file pipeline run
# to one `P1050349_face0.jpg` doing exactly that).
_GENERATED_CROP_PATTERN = re.compile(r"_face\d+\.(jpg|jpeg|png)$", re.IGNORECASE)

# Subtrees inside an Apple photo library (modern .photoslibrary
# bundles AND older iPhoto Library directories) that hold
# auto-generated previews / metadata / face crops / per-album caches.
# We keep `originals/` (modern) and `Originals/` (iPhoto) so the real
# photos still get indexed.
_PHOTO_LIBRARY_SKIP_DIRS = frozenset({
    # modern .photoslibrary
    "Thumbnails", "resources", "private", "external",
    "database", "scopes",
    # iPhoto Library
    "Data", "Faces", "Caches", "iPod Photo Cache",
    "Auto Import", "Trash", "Auto Save",
    "Library6.iPhoto", "ProjectDBVersion",
})


def _is_inside_photo_library(dirpath: str) -> bool:
    """True if dirpath is anywhere under an Apple photo library bundle."""
    lower = dirpath.lower()
    return ".photoslibrary" in lower or "iphoto library" in lower


def delete_file_row(conn, file_id: int) -> None:
    """Delete a files row plus the sqlite-vec virtual-table rows that
    won't be reached by CASCADE.

    ``faces`` and ``detections`` go away via the foreign-key cascade
    on ``files``, but ``vec_clip``, ``vec_clip_frames`` and
    ``vec_faces`` are virtual tables — no FK, no cascade. A bare
    ``DELETE FROM files`` orphans their rowids, and a future face/
    embedding INSERT eventually picks one of those rowids and dies
    with ``UNIQUE constraint failed``.

    Use this helper everywhere a files row is deleted. Idempotent if
    the row is already gone.
    """
    conn.execute("DELETE FROM vec_clip WHERE rowid=?", (file_id,))
    for fr in conn.execute(
        "SELECT id FROM frames WHERE file_id=?", (file_id,)
    ).fetchall():
        conn.execute("DELETE FROM vec_clip_frames WHERE rowid=?", (fr[0],))
    for fa in conn.execute(
        "SELECT id FROM faces WHERE file_id=?", (file_id,)
    ).fetchall():
        conn.execute("DELETE FROM vec_faces WHERE rowid=?", (fa[0],))
    conn.execute("DELETE FROM files WHERE id=?", (file_id,))


def _walk_one_root(root: Path) -> Iterator[Path]:
    """Yield supported image paths under a single root.

    No per-file open here — enumeration must stay cheap so incremental
    scans finish quickly. The dimension-based "is this a thumbnail?"
    check happens in ``scan()`` for new/changed files only.

    Skipped subtrees inside `.photoslibrary` and `iPhoto Library`
    bundles: Thumbnails / Faces / Data / resources / etc. — auto-
    generated previews and metadata. Real photos live in `originals/`
    (modern) or `Originals/` (iPhoto), which we keep. Per-face crop
    files (`*_face0.jpg`) are also skipped — they crash InsightFace
    when fed back through it.
    """
    root = root.expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXTS:
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if _is_inside_photo_library(dirpath):
            dirnames[:] = [
                d for d in dirnames if d not in _PHOTO_LIBRARY_SKIP_DIRS
            ]
        for fn in filenames:
            if fn.startswith("."):
                continue
            if _GENERATED_CROP_PATTERN.search(fn):
                continue
            if Path(fn).suffix.lower() in SUPPORTED_EXTS:
                yield Path(dirpath) / fn


def discover(roots: list[Path]) -> Iterator[Path]:
    """Serial walker over multiple roots. Kept for callers / tests that
    want a simple flat iterator (parallel walking lives in ``scan``).
    """
    for root in roots:
        yield from _walk_one_root(root)


def scan(
    roots: list[Path],
    conn: "db.sqlite3.Connection",
    prune: bool = False,
    min_pixels: int = MIN_PIXELS_DEFAULT,
    dedupe: bool = True,
    walk_workers: int | None = None,
) -> dict[str, int]:
    """Walk *roots*, insert/update `files`, return summary counts.

    Each root is walked in its own thread (capped by ``walk_workers``,
    default ``min(len(roots), 8)``) and discovered paths are streamed
    into the main thread for processing. Big win on SMB/network mounts
    where directory enumeration latency dominates wall time —
    overlapping walks across roots hides per-root stat latency. Within
    each root the walk is still serial (single-threaded ``os.walk``);
    SMB servers with NVMe caches handle concurrent reads from
    *different* mounts much better than concurrent reads from the same
    one, so per-root threading is the right granularity.

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

    if walk_workers is None:
        walk_workers = min(max(len(roots), 1), 8)

    # Bounded queue: walkers block when consumer falls behind so a slow
    # SQLite write (changed/new file branch) doesn't let memory blow up
    # with millions of pending Path objects.
    path_q: queue.Queue = queue.Queue(maxsize=10_000)
    SENTINEL = (None, None)

    def walker(root_idx: int, root: Path) -> None:
        try:
            for p in _walk_one_root(root):
                path_q.put((root_idx, p))
        except Exception as e:
            # Surface walker failures as a special error tuple so we can
            # bump stats without crashing the consumer.
            path_q.put((root_idx, e))
        finally:
            path_q.put((root_idx, None))  # per-root done marker

    console = Console(stderr=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        # One bar per root so the user sees which network mount is the
        # current bottleneck. Truncate long paths from the front so the
        # leaf directory (the part that matters) stays readable.
        def _short(p: Path) -> str:
            s = str(p)
            return s if len(s) <= 60 else "…" + s[-59:]

        per_root = [
            prog.add_task(f"  {_short(r)}", total=None, start=True)
            for r in roots
        ]
        overall = prog.add_task("[bold]processing[/bold]", total=None)
        per_root_count = [0] * len(roots)
        per_root_done = [False] * len(roots)

        pool = ThreadPoolExecutor(
            max_workers=walk_workers, thread_name_prefix="walk"
        )
        for i, r in enumerate(roots):
            pool.submit(walker, i, r)

        active = len(roots)
        processed = 0
        try:
            while active > 0:
                root_idx, payload = path_q.get()
                if payload is None:
                    # Walker for this root finished.
                    active -= 1
                    per_root_done[root_idx] = True
                    prog.update(
                        per_root[root_idx],
                        description=f"  ✓ {_short(roots[root_idx])} "
                                    f"({per_root_count[root_idx]} files)",
                    )
                    continue
                if isinstance(payload, Exception):
                    stats["errors"] += 1
                    log_msg = f"walker error in {roots[root_idx]}: {payload}"
                    console.print(f"[red]{log_msg}[/red]")
                    continue

                path: Path = payload
                per_root_count[root_idx] += 1
                prog.update(
                    per_root[root_idx],
                    completed=per_root_count[root_idx],
                    description=f"  {_short(roots[root_idx])} "
                                f"({per_root_count[root_idx]} files)",
                )
                processed += 1
                prog.update(
                    overall, completed=processed,
                    description=f"[bold]processing[/bold] · {processed} seen",
                )

                spath = str(path)
                # Dedupe within a single scan. With overlapping roots
                # (e.g. /Volumes/photo and /Volumes/photo/107_PANA) the
                # parallel walkers will yield the same path multiple
                # times. Without this guard the second visit sees
                # cached=None (the `known` dict was loaded once at the
                # start) and tries to re-INSERT, hitting the path
                # UNIQUE constraint.
                if spath in paths_seen:
                    continue
                paths_seen.add(spath)

                # Wrap the whole per-path body so a single broken file
                # never aborts a multi-hour scan. We've burnt entire
                # 250k-file runs on a single IntegrityError before.
                try:
                    _process_one_path(
                        path, spath, conn, known,
                        paths_seen, stats, min_pixels, dedupe,
                    )
                except Exception as e:
                    log.warning("scan failed for %s: %s", spath, e)
                    stats["errors"] += 1
        finally:
            pool.shutdown(wait=True)

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


def _process_one_path(
    path: Path,
    spath: str,
    conn,
    known: dict,
    paths_seen: set,
    stats: dict,
    min_pixels: int,
    dedupe: bool,
) -> None:
    """One-file body of the scan loop.

    Pulled out of ``scan()`` so a per-file failure (filesystem race,
    integrity hiccup) can be caught without killing a multi-hour
    scan. Updates to ``stats`` / ``paths_seen`` / ``known`` happen
    in place; ``known`` is updated after every successful INSERT/
    UPDATE so a duplicate emission of the same path within a single
    scan (e.g. from overlapping roots) takes the fast path on its
    second visit.
    """
    try:
        st = path.stat()
    except OSError:
        stats["errors"] += 1
        return

    cached = known.get(spath)
    now = time.time()
    mime = mimetypes.guess_type(spath)[0]

    # Fast path: file is known and unchanged. No PIL open, no hash,
    # no SQL — just count and return. Dominant case on incremental
    # scans.
    if cached is not None:
        m, sz, missing, too_small, fid = cached
        if m == st.st_mtime and sz == st.st_size:
            if missing:
                conn.execute("UPDATE files SET missing=0 WHERE id=?", (fid,))
                known[spath] = (m, sz, 0, too_small, fid)
            if too_small:
                stats["skipped_small"] += 1
            else:
                stats["unchanged"] += 1
            return

        # Changed — re-apply size filter and re-hash.
        if _is_too_small(path, min_pixels):
            conn.execute(
                """UPDATE files SET size=?, mtime=?, too_small=1 WHERE id=?""",
                (st.st_size, st.st_mtime, fid),
            )
            known[spath] = (st.st_mtime, st.st_size, 0, 1, fid)
            stats["skipped_small"] += 1
            return
        try:
            chash = content_hash(path)
        except OSError:
            stats["errors"] += 1
            return
        conn.execute(
            """UPDATE files SET content_hash=?, size=?, mtime=?, mime=?,
                   indexed_at=?, meta_done=0, yolo_done=0, faces_done=0,
                   clip_done=0, missing=0, too_small=0
               WHERE id=?""",
            (chash, st.st_size, st.st_mtime, mime, now, fid),
        )
        known[spath] = (st.st_mtime, st.st_size, 0, 0, fid)
        stats["changed"] += 1
        return

    # New file. Apply size filter; tombstone too-small entries so
    # the next scan takes the fast path with no PIL header read.
    if _is_too_small(path, min_pixels):
        cur = conn.execute(
            """INSERT INTO files
               (path, content_hash, size, mtime, mime, indexed_at, too_small)
               VALUES (?, '', ?, ?, ?, ?, 1)""",
            (spath, st.st_size, st.st_mtime, mime, now),
        )
        known[spath] = (st.st_mtime, st.st_size, 0, 1, cur.lastrowid)
        stats["skipped_small"] += 1
        return
    try:
        chash = content_hash(path)
    except OSError:
        stats["errors"] += 1
        return

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
            paths_seen.add(live["path"])
            stats["dedup_skipped"] += 1
            return

        dead = conn.execute(
            """SELECT id FROM files
               WHERE content_hash = ? AND missing = 1
               LIMIT 1""",
            (chash,),
        ).fetchone()
        if dead is not None:
            conn.execute(
                """UPDATE files
                   SET path=?, size=?, mtime=?, mime=?,
                       missing=0, indexed_at=?
                   WHERE id=?""",
                (spath, st.st_size, st.st_mtime, mime, now, dead["id"]),
            )
            known[spath] = (st.st_mtime, st.st_size, 0, 0, dead["id"])
            stats["moved"] += 1
            return

    cur = conn.execute(
        """INSERT INTO files
           (path, content_hash, size, mtime, mime, indexed_at, kind)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (spath, chash, st.st_size, st.st_mtime, mime, now,
         kind_for_ext(path.suffix)),
    )
    known[spath] = (st.st_mtime, st.st_size, 0, 0, cur.lastrowid)
    stats["new"] += 1


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
        walk_workers: int | None = typer.Option(
            None, "--walk-workers",
            help="Threads for the parallel directory walk (default: "
                 "min(roots, 8)). Useful to raise on multi-mount NAS "
                 "setups where SMB latency dominates.",
        ),
    ) -> None:
        """Scan directories for images/videos and populate the file index."""
        cfg = config.load()
        db.init(cfg.db_path)
        conn = db.connect(cfg.db_path)
        try:
            result = scan(paths, conn, prune=prune,
                          min_pixels=min_pixels, dedupe=dedupe,
                          walk_workers=walk_workers)
            console = Console()
            for k, v in result.items():
                console.print(f"  {k}: {v}")
        finally:
            conn.close()

    @parent.command(name="rescan")
    def cmd_rescan(
        prune: bool = typer.Option(
            True, "--prune/--no-prune",
            help="Mark files not found on disk as missing (default: yes).",
        ),
        min_pixels: int = typer.Option(
            MIN_PIXELS_DEFAULT, "--min-pixels",
            help="Skip images where BOTH dimensions are below this.",
        ),
        walk_workers: int | None = typer.Option(
            None, "--walk-workers",
            help="Threads for the parallel directory walk (default: "
                 "min(roots, 8)).",
        ),
    ) -> None:
        """Re-scan every directory previously passed to ``scan``.

        Reads the ``scan_roots`` table — populated automatically by the
        ``scan`` command — and walks each root again. Use this after
        you've added or removed photos in any of your known locations
        and want to refresh the index in one shot, without retyping
        every path.
        """
        cfg = config.load()
        db.init(cfg.db_path)
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            roots = [
                Path(r[0]) for r in conn.execute(
                    "SELECT path FROM scan_roots ORDER BY path"
                )
            ]
            if not roots:
                console.print(
                    "[yellow]no scan_roots recorded[/yellow] — "
                    "run `image-wizard scan <path>` at least once first."
                )
                return
            console.print(
                f"[bold]rescanning {len(roots)} root(s):[/bold]"
            )
            for r in roots:
                exists = "" if r.exists() else "  [red](unmounted)[/red]"
                console.print(f"  • {r}{exists}")
            # Drop unmounted roots so we don't false-positive --prune
            # files under a temporarily-unavailable network mount.
            roots = [r for r in roots if r.exists()]
            if not roots:
                console.print("[red]no roots are currently mounted/accessible[/red]")
                return
            result = scan(roots, conn, prune=prune, min_pixels=min_pixels,
                          walk_workers=walk_workers)
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
                    delete_file_row(conn, r["id"])
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
                            # CASCADE cleans detections / faces, but not the
                            # sqlite-vec virtual tables — delete_file_row
                            # handles those manually.
                            delete_file_row(conn, r["id"])
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

        Mostly historical: V1 video support re-enabled .mov/.mp4 indexing,
        so this command is now an opt-out for users who don't want videos
        indexed. CASCADE handles detections / faces; the helper also
        cleans the sqlite-vec virtual tables so we don't leave orphan
        embedding rows behind.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        try:
            like = " OR ".join(["LOWER(path) LIKE ?"] * len(VIDEO_EXTS))
            params = [f"%{e}" for e in VIDEO_EXTS]
            ids = [
                r[0] for r in conn.execute(
                    f"SELECT id FROM files WHERE {like}", params
                ).fetchall()
            ]
            for fid in ids:
                delete_file_row(conn, fid)
            conn.commit()
            Console().print(f"  dropped: {len(ids)} video rows")
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
        kernel: bool = typer.Option(
            False, "--kernel",
            help="Also dump recent macOS kernel log entries mentioning "
                 "Python / image-wizard (slow — runs `log show`).",
        ),
    ) -> None:
        """Show the tail of the index checkpoint log + faulthandler dump.

        Use this after a silent crash (segfault, OOM kill, supervisor
        kill) to see exactly which file/stage was active at the moment
        of death and any native-code traceback that fired.

        Files inspected:
          <cache>/logs/index.log         per-file lifecycle + memory snapshots
          <cache>/logs/faulthandler.log  Python frames at SIGSEGV/SIGABRT/SIGTERM

        Lines in index.log:
          <unix_ts> start    <file_id> <path>
          <unix_ts> stage    <file_id> {yolo|clip|faces}
          <unix_ts> done     <file_id>
          <unix_ts> error    <file_id> <message>
          <unix_ts> mem      processed=N rss_mb=M
          <unix_ts> throttle rss_mb=M ceiling_mb=C
        """
        import subprocess
        cfg = config.load()
        log_dir = cfg.cache_dir / "logs"
        console = Console()

        index_log = log_dir / "index.log"
        if not index_log.exists():
            console.print(f"[yellow]no log at {index_log}[/yellow]")
            console.print("Run `image-wizard index` first.")
            return

        console.print(f"[bold]checkpoint log:[/bold] [dim]{index_log}[/dim]")
        with index_log.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, 64 * 1024)
            f.seek(size - block)
            chunk = f.read().decode("utf-8", errors="replace")
        lines = chunk.splitlines()[-tail:]
        for ln in lines:
            if " error " in ln:
                console.print(f"[red]{ln}[/red]")
            elif " stage " in ln:
                console.print(f"[magenta]{ln}[/magenta]")
            elif " mem " in ln or " throttle " in ln:
                console.print(f"[cyan]{ln}[/cyan]")
            elif ln.startswith(("---",)) or " ---" in ln:
                console.print(f"[bold yellow]{ln}[/bold yellow]")
            else:
                console.print(ln)

        fh_log = log_dir / "faulthandler.log"
        if fh_log.exists() and fh_log.stat().st_size > 0:
            console.print(f"\n[bold]faulthandler dump:[/bold] [dim]{fh_log}[/dim]")
            console.print("[yellow](native-code traceback — last 100 lines)[/yellow]")
            text = fh_log.read_text(errors="replace").splitlines()[-100:]
            for ln in text:
                console.print(ln)
        else:
            console.print(
                "\n[dim](no faulthandler.log — process wasn't terminated by "
                "a signal Python could trap)[/dim]"
            )

        if kernel:
            console.print("\n[bold]kernel events (last hour):[/bold]")
            try:
                pred = (
                    'eventMessage CONTAINS[c] "image-wizard" '
                    'OR eventMessage CONTAINS[c] "imagewizard" '
                    'OR (process == "kernel" AND eventMessage CONTAINS "memorystatus") '
                    'OR (process == "kernel" AND eventMessage CONTAINS "Jetsam")'
                )
                out = subprocess.run(
                    ["log", "show", "--last", "1h", "--predicate", pred],
                    capture_output=True, text=True, timeout=30,
                )
                lines = (out.stdout or "").splitlines()
                if not lines:
                    console.print("[dim]no relevant kernel events[/dim]")
                for ln in lines[-50:]:
                    console.print(ln)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                console.print(f"[red]log show failed: {e}[/red]")

    @parent.command(name="list-videos")
    def cmd_list_videos(
        list_paths: bool = typer.Option(
            False, "--list",
            help="Print every video path. Default: just show the summary.",
        ),
        state: str = typer.Option(
            "all", "--state",
            help="Filter --list output: all | indexed | pending | failed | unknown.",
        ),
    ) -> None:
        """Survey video files across every scanned root.

        Walks each root in ``scan_roots`` looking for video extensions
        (.mov / .mp4 / .m4v / .avi / .mkv) and reports each file's state
        in the DB:

          • indexed   — has run through the ML pipeline (yolo_done=1)
          • pending   — known to scan but not yet indexed
          • failed    — tombstoned with decode_failed=1 (run
                        ``clear-failures`` to retry)
          • unknown   — present on disk, never seen by ``scan``
                        (run ``rescan`` to pick them up)

        Useful before kicking off a big ``index`` run on a fresh V2
        deploy, or to spot-check what the next ``rescan`` will add.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            roots = [
                Path(r[0]) for r in conn.execute(
                    "SELECT path FROM scan_roots ORDER BY path"
                )
            ]
            if not roots:
                console.print("[yellow]no scan_roots recorded[/yellow]")
                return

            # Pull every known video row into a dict for O(1) per-path lookup.
            known: dict[str, str] = {}
            for r in conn.execute(
                """SELECT path, decode_failed, yolo_done
                   FROM files
                   WHERE missing=0 AND kind='video'"""
            ):
                if r["decode_failed"]:
                    known[r["path"]] = "failed"
                elif r["yolo_done"]:
                    known[r["path"]] = "indexed"
                else:
                    known[r["path"]] = "pending"

            # Walk each root for .mov/.mp4/etc on disk.
            on_disk: list[tuple[Path, str]] = []
            for root in roots:
                if not root.exists():
                    console.print(f"[dim]skipping unmounted root: {root}[/dim]")
                    continue
                console.print(f"[dim]scanning {root}…[/dim]")
                for p in _walk_one_root(root):
                    if p.suffix.lower() in VIDEO_EXTS:
                        sp = str(p)
                        on_disk.append((p, known.get(sp, "unknown")))

            counts: dict[str, int] = {
                "indexed": 0, "pending": 0, "failed": 0, "unknown": 0,
            }
            for _, st in on_disk:
                counts[st] = counts.get(st, 0) + 1

            console.print()
            console.print(f"[bold]{len(on_disk)} video files across {len(roots)} root(s)[/bold]")
            for k in ("indexed", "pending", "failed", "unknown"):
                colour = {"indexed": "green", "pending": "yellow",
                          "failed": "red", "unknown": "cyan"}[k]
                console.print(f"  [{colour}]{k:9s}[/{colour}] {counts[k]}")

            if list_paths:
                console.print()
                shown = 0
                for p, st in sorted(on_disk):
                    if state != "all" and st != state:
                        continue
                    colour = {"indexed": "green", "pending": "yellow",
                              "failed": "red", "unknown": "cyan"}[st]
                    console.print(f"  [{colour}]{st:9s}[/{colour}] {p}")
                    shown += 1
                if shown == 0:
                    console.print(f"[dim]no videos matched --state={state}[/dim]")
        finally:
            conn.close()

    @parent.command(name="skip")
    def cmd_skip(
        target: str = typer.Argument(
            ...,
            help="File id (integer), content hash, or path/path-substring.",
        ),
        reason: str = typer.Option(
            "manually skipped",
            "--reason",
            help="Stored in decode_error so you remember why later.",
        ),
    ) -> None:
        """Tombstone one file so the pipeline skips it.

        Useful when a single file crashes the indexer and you want to
        keep going. The file stays in the DB but every ML stage skips
        it. Reverse with `clear-failures`.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            row = None
            if target.isdigit():
                row = conn.execute(
                    "SELECT id, path FROM files WHERE id=?", (int(target),)
                ).fetchone()
            if row is None and len(target) == 64:
                row = conn.execute(
                    "SELECT id, path FROM files WHERE content_hash=?", (target,)
                ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT id, path FROM files WHERE path = ? OR path LIKE ? LIMIT 1",
                    (target, f"%{target}%"),
                ).fetchone()
            if row is None:
                console.print(f"[red]no file matches {target!r}[/red]")
                raise typer.Exit(1)
            conn.execute(
                "UPDATE files SET decode_failed=1, decode_error=? WHERE id=?",
                (reason[:500], row["id"]),
            )
            console.print(
                f"[green]skipped[/green] file #{row['id']}: {row['path']}"
            )
            console.print(f"  reason: {reason}")
            console.print("Run `image-wizard clear-failures` to undo.")
        finally:
            conn.close()

    @parent.command(name="purge-orphans")
    def cmd_purge_orphans(
        dry_run: bool = typer.Option(
            False, "--dry-run",
            help="Count orphans without changing the DB.",
        ),
    ) -> None:
        """Delete orphan rows from sqlite-vec virtual tables.

        ``vec_clip``, ``vec_clip_frames`` and ``vec_faces`` are
        virtual tables and don't get touched by ``ON DELETE CASCADE``
        — when a ``files`` row is deleted (e.g. by ``cleanup-thumbnails``,
        ``drop-small``, ``find-duplicates --delete``) the matching
        face rows go away via cascade but the corresponding vector
        rows stay behind. As ``faces.id`` autoincrement walks past
        the orphan rowids, new INSERTs collide with them →
        ``UNIQUE constraint failed on vec_faces primary key``.

        This sweeps the orphans. Cheap and safe to run any time;
        the pipeline now also DELETE-before-INSERTs each rowid so
        new orphans don't accumulate.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            # vec_clip rowid = files.id
            vec_clip_orphans = conn.execute(
                """SELECT rowid FROM vec_clip
                   WHERE rowid NOT IN (SELECT id FROM files)"""
            ).fetchall()
            # vec_faces rowid = faces.id
            vec_faces_orphans = conn.execute(
                """SELECT rowid FROM vec_faces
                   WHERE rowid NOT IN (SELECT id FROM faces)"""
            ).fetchall()
            # vec_clip_frames rowid = frames.id
            vec_frames_orphans = conn.execute(
                """SELECT rowid FROM vec_clip_frames
                   WHERE rowid NOT IN (SELECT id FROM frames)"""
            ).fetchall()

            console.print(
                f"[bold]orphan vector rows:[/bold] "
                f"vec_clip={len(vec_clip_orphans)}, "
                f"vec_faces={len(vec_faces_orphans)}, "
                f"vec_clip_frames={len(vec_frames_orphans)}"
            )
            if dry_run:
                console.print("[yellow]dry-run — nothing deleted[/yellow]")
                return

            for r in vec_clip_orphans:
                conn.execute("DELETE FROM vec_clip WHERE rowid=?", (r[0],))
            for r in vec_faces_orphans:
                conn.execute("DELETE FROM vec_faces WHERE rowid=?", (r[0],))
            for r in vec_frames_orphans:
                conn.execute("DELETE FROM vec_clip_frames WHERE rowid=?", (r[0],))
            console.print(
                f"[green]deleted {len(vec_clip_orphans) + len(vec_faces_orphans) + len(vec_frames_orphans)}"
                f" orphan row(s)[/green]"
            )
        finally:
            conn.close()

    @parent.command(name="cleanup-thumbnails")
    def cmd_cleanup_thumbnails(
        dry_run: bool = typer.Option(
            False, "--dry-run",
            help="Show what would be removed without changing the DB.",
        ),
    ) -> None:
        """Remove auto-generated photo-library thumbnails from the index.

        Older scans (before iPhoto Library exclusion landed) indexed
        files under `iPhoto Library/Thumbnails/...` and per-face crop
        files (`*_face0.jpg`). They're auto-generated, useless for
        search, and one of them just crashed an entire 7000-file index
        run. This drops them from `files`; CASCADEs handle detections /
        faces / vec_clip rows.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            rows = conn.execute(
                """SELECT id, path FROM files
                   WHERE path LIKE '%iPhoto Library/Thumbnails/%'
                      OR path LIKE '%.photoslibrary/Thumbnails/%'
                      OR path LIKE '%.photoslibrary/resources/%'
                      OR path LIKE '%_face0.jpg'
                      OR path LIKE '%_face0.jpeg'
                      OR path LIKE '%_face1.jpg'
                      OR path LIKE '%_face2.jpg'
                      OR path LIKE '%_face3.jpg'"""
            ).fetchall()
            console.print(f"[bold]{len(rows)} candidate row(s)[/bold]")
            for r in rows[:20]:
                console.print(f"  #{r['id']:7d}  {r['path']}")
            if len(rows) > 20:
                console.print(f"  ... and {len(rows) - 20} more")
            if dry_run:
                console.print("[yellow]dry-run — nothing deleted[/yellow]")
                return
            if not rows:
                return
            for r in rows:
                delete_file_row(conn, r["id"])
            console.print(f"[green]deleted {len(rows)} row(s)[/green]")
        finally:
            conn.close()

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
