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
    "Previews",  # rendered preview derivatives — not originals
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


def _find_last_in_flight(log_path: "Path") -> tuple[int, str] | None:
    """Scan the tail of the checkpoint log and return the (file_id, path)
    of the file that was in flight when the process died.

    The pipeline is single-threaded on the consumer side, so at any
    moment at most one file is between its ``start`` and ``done``
    lines. The in-flight file is the one with the last ``start`` that
    has no corresponding ``done`` after it. Returns None when the log
    is missing or the tail shows every started file completed cleanly
    (a rare "process exited after done, before next start" case that
    we won't confidently pin on any single file).
    """
    if not log_path.exists():
        return None
    # 512 KB tail is plenty — the pipeline logs a few hundred lines per
    # minute at most, and we only care about the recent state.
    with log_path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = min(size, 512 * 1024)
        f.seek(size - block)
        chunk = f.read().decode("utf-8", errors="replace")

    last_start: tuple[int, str] | None = None
    for line in chunk.splitlines():
        # Format: "<unix_ts> <event> <args...>"
        parts = line.split(maxsplit=3)
        if len(parts) < 3:
            continue
        event = parts[1]
        try:
            fid = int(parts[2])
        except ValueError:
            continue
        if event == "start":
            path = parts[3] if len(parts) > 3 else ""
            last_start = (fid, path)
        elif event == "done" and last_start and last_start[0] == fid:
            # The last-started file completed. Reset so we only pin
            # blame on a start that has *no* subsequent done.
            last_start = None
    return last_start


def _live_photo_still_exts() -> list[str]:
    """Extensions we treat as the still half of a Live Photo pair."""
    return [".heic", ".heif", ".jpg", ".jpeg", ".png"]


def detect_live_photos(conn, only_new: bool = True) -> int:
    """Flag video files whose basename matches a still-image sibling.

    iPhone Live Photos store as ``IMG_1234.HEIC`` (the still) plus
    ``IMG_1234.MOV`` (1–2s of motion around the shot). Both land in
    ``files`` and the .MOV shows up in the UI as a phantom duplicate
    video. Detect the pairing by basename equality within the same
    directory and set ``files.live_photo_of`` on the video to point
    at the still's id.

    Idempotent: with ``only_new`` (default) we skip videos that
    already have ``live_photo_of`` set. Pass ``only_new=False`` to
    re-scan everything (e.g. after fixing a bad match).

    Returns the number of files newly flagged.
    """
    where = "missing = 0"
    if only_new:
        where += " AND live_photo_of IS NULL"
    videos = conn.execute(
        f"""SELECT id, path FROM files
            WHERE {where}
              AND (
                LOWER(SUBSTR(path, -4)) IN ('.mov', '.mp4')
                OR LOWER(SUBSTR(path, -4)) = '.m4v'
              )"""
    ).fetchall()

    still_exts = _live_photo_still_exts()
    flagged = 0
    for r in videos:
        p = Path(r["path"])
        prefix = str(p.parent) + os.sep + p.stem + "."
        # Look up any still-image file in the same directory with the
        # matching basename. Multiple candidate rows (HEIC + JPG both)
        # are fine — pick the lowest id for determinism.
        candidates: list[int] = []
        for ext in still_exts:
            row = conn.execute(
                "SELECT id FROM files WHERE missing=0 AND path=? LIMIT 1",
                (prefix + ext[1:],),  # ext already has leading '.'
            ).fetchone() or conn.execute(
                # Case-insensitive fallback for weird filesystems
                """SELECT id FROM files
                   WHERE missing=0
                     AND LOWER(path) = LOWER(?)
                   LIMIT 1""",
                (prefix + ext[1:],),
            ).fetchone()
            if row:
                candidates.append(row["id"])
        if candidates:
            still_id = min(candidates)
            conn.execute(
                "UPDATE files SET live_photo_of=? WHERE id=?",
                (still_id, r["id"]),
            )
            flagged += 1

    # Reconcile the reverse links: every still that a .MOV points at
    # gets live_video_id set back to that .MOV. Runs unconditionally so
    # it also backfills pairs detected before the live_video_id column
    # existed. Clear stale reverse links first so a since-deleted .MOV
    # doesn't leave a dangling badge.
    conn.execute("UPDATE files SET live_video_id=NULL")
    conn.execute(
        """UPDATE files SET live_video_id = (
               SELECT mov.id FROM files mov
               WHERE mov.live_photo_of = files.id AND mov.missing = 0
               ORDER BY mov.id LIMIT 1
           )
           WHERE id IN (
               SELECT DISTINCT live_photo_of FROM files
               WHERE live_photo_of IS NOT NULL AND missing = 0
           )"""
    )
    return flagged


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
    conn.execute("DELETE FROM ocr_fts WHERE rowid=?", (file_id,))
    conn.execute("DELETE FROM files WHERE id=?", (file_id,))


def _flush_ocr(conn, pending: list) -> None:
    """Write a batch of (file_id, text) OCR results to the FTS index.

    FTS5 has no UPSERT, so we delete any prior row for the file first
    (matters for --redo). Sets ocr_done=1 regardless of whether text was
    found, so empty results aren't retried every run.
    """
    for fid, text in pending:
        conn.execute("DELETE FROM ocr_fts WHERE rowid=?", (fid,))
        if text and text.strip():
            conn.execute(
                "INSERT INTO ocr_fts (rowid, text) VALUES (?, ?)", (fid, text)
            )
        conn.execute("UPDATE files SET ocr_done=1 WHERE id=?", (fid,))
    conn.commit()


def compute_phashes(conn, cache_dir, console=None, workers: int = 4) -> int:
    """Fill ``files.phash`` for live files that don't have one yet.

    The perceptual (DCT) hash is computed from the file's 512px cached
    thumbnail — fast, and robust to JPEG recompression, so two
    byte-different exports of the same photo get the same phash. Cached
    in the DB so this only pays the cost once. Returns how many were
    computed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import imagehash
    from PIL import Image
    from .thumbs import thumb_path, ensure_thumbnail

    # Photos only — perceptual hashing is for finding duplicate images.
    # Videos would force a poster-frame decode (slow, and corrupt .MOVs
    # spew ffmpeg errors); byte-identical duplicate videos are still
    # caught by the default content_hash mode.
    rows = conn.execute(
        """SELECT id, content_hash, path FROM files
           WHERE missing = 0 AND too_small = 0 AND phash IS NULL
             AND dup_of IS NULL AND content_hash != ''
             AND COALESCE(kind, 'image') != 'video'"""
    ).fetchall()
    if not rows:
        return 0

    def one(r):
        try:
            tp = thumb_path(cache_dir, r["content_hash"])
            if not tp.exists():
                from . import decode
                src = Path(r["path"])
                if not src.exists():
                    return (r["id"], None)
                if src.suffix.lower() in VIDEO_EXTS:
                    from .video import extract_poster
                    arr, _ = extract_poster(src)
                else:
                    arr = decode.load_image(src)
                ensure_thumbnail(arr, cache_dir, r["content_hash"])
            with Image.open(tp) as im:
                return (r["id"], str(imagehash.phash(im.convert("RGB"))))
        except Exception:
            return (r["id"], None)

    done = 0
    from rich.progress import Progress, BarColumn, MofNCompleteColumn, SpinnerColumn, TimeRemainingColumn
    prog = Progress(
        SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
        console=console or Console(stderr=True),
    )
    with prog, ThreadPoolExecutor(max_workers=workers) as pool:
        task = prog.add_task("phash", total=len(rows))
        pending = []
        futs = {pool.submit(one, r): r for r in rows}
        for fut in as_completed(futs):
            fid, ph = fut.result()
            # Store '' for un-hashable files so we don't retry them forever.
            pending.append((ph if ph is not None else "", fid))
            done += 1
            prog.advance(task)
            if len(pending) >= 500:
                conn.executemany("UPDATE files SET phash=? WHERE id=?", pending)
                conn.commit()
                pending = []
        if pending:
            conn.executemany("UPDATE files SET phash=? WHERE id=?", pending)
            conn.commit()
    return done


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
        "live_photo_pairs": 0,
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

    # Detect iPhone Live Photo pairs (HEIC + MOV with matching basenames).
    # Runs after the walk so both halves of any newly-added pair are in
    # the DB before we try to match them. Cheap: one indexed SELECT per
    # video row that doesn't already have live_photo_of set.
    stats["live_photo_pairs"] = detect_live_photos(conn, only_new=True)

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
            help="Delete redundant files from disk AND drop their index rows.",
        ),
        dedupe_index: bool = typer.Option(
            False, "--dedupe-index",
            help="Drop redundant index rows but KEEP the files on disk. "
                 "Non-destructive fix for 'timeline shows the same photo "
                 "twice' — collapses each duplicate group to one entry. "
                 "Durable: a later rescan re-skips the kept-on-disk copies "
                 "via hash dedup.",
        ),
        visual: bool = typer.Option(
            False, "--visual",
            help="Group by perceptual (visual) hash instead of exact "
                 "bytes. Catches re-encoded / re-imported copies that are "
                 "visually identical but byte-different (so content_hash "
                 "differs). Computes + caches a phash per photo on first "
                 "run (slow once, instant after).",
        ),
        reset: bool = typer.Option(
            False, "--reset",
            help="Un-hide everything previously marked by --dedupe-index "
                 "(clear all dup_of flags) and exit. Reverses a dedup.",
        ),
        keep: str = typer.Option(
            "shortest-path", "--keep",
            help="Tie-break for which copy to keep per group: "
                 "shortest-path | longest-path | oldest | newest. "
                 "(A copy carrying GPS or a date always wins over one "
                 "without, regardless of this.)",
        ),
        limit: int = typer.Option(
            0, "--limit",
            help="Stop after processing this many duplicate groups (0 = all).",
        ),
        verbose: bool = typer.Option(
            False, "--verbose", help="Print every group (slow for thousands)."
        ),
    ) -> None:
        """Find and resolve duplicate photos.

        Groups either by exact bytes (content_hash, default) or by
        perceptual hash (--visual — catches re-encoded/re-imported
        copies that look identical but differ byte-for-byte). One file
        per group is the keeper; the rest are redundant.

        * default — dry run, just report the counts.
        * --dedupe-index — hide the redundant copies from all views by
          marking them dup_of the keeper. Non-destructive (files + rows
          stay) and durable: a rescan leaves the flag alone, so it works
          for visual duplicates too. Reverse with --reset.
        * --delete — unlink the redundant files from disk and drop their
          rows (reclaims space; not reversible).
        * --reset — clear all dup_of flags (un-hide everything).

        Keeper selection preserves metadata: a copy with GPS beats one
        without, then a copy with a date, then the --keep tie-break.
        """
        if keep not in ("shortest-path", "longest-path", "oldest", "newest"):
            raise typer.BadParameter(
                "--keep must be one of: shortest-path, longest-path, oldest, newest"
            )
        if delete and dedupe_index:
            raise typer.BadParameter("pass only one of --delete / --dedupe-index")

        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            if reset:
                cur = conn.execute(
                    "UPDATE files SET dup_of=NULL WHERE dup_of IS NOT NULL"
                )
                console.print(
                    f"[green]un-hid {cur.rowcount} duplicate(s)[/green] "
                    "(all dup_of flags cleared)"
                )
                return

            # Grouping key: exact content_hash (default) or perceptual
            # phash (--visual, for byte-different visual duplicates).
            key_col = "phash" if visual else "content_hash"
            # --visual is photos-only (a video poster-frame phash could
            # spuriously collide); the default byte mode covers videos.
            extra = "AND COALESCE(kind,'image') != 'video'" if visual else ""
            if visual:
                n = compute_phashes(conn, cfg.cache_dir, console)
                if n:
                    console.print(f"[dim]computed {n} perceptual hashes[/dim]")

            hashes = conn.execute(
                f"""SELECT {key_col} AS gkey, COUNT(*) AS cnt
                    FROM files
                    WHERE missing = 0 AND too_small = 0 AND dup_of IS NULL
                      AND {key_col} IS NOT NULL AND {key_col} != '' {extra}
                    GROUP BY {key_col}
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
            if not delete and not dedupe_index:
                console.print(
                    "[dim]dry run — pass --dedupe-index (keep files) or "
                    "--delete (remove files)[/dim]"
                )

            def _tiebreak(rows):
                if keep == "shortest-path":
                    return min(rows, key=lambda r: (len(r["path"]), r["path"]))
                if keep == "longest-path":
                    return max(rows, key=lambda r: (len(r["path"]), r["path"]))
                if keep == "oldest":
                    return min(rows, key=lambda r: r["mtime"])
                return max(rows, key=lambda r: r["mtime"])

            def _picker(rows):
                # Preserve metadata: prefer copies with GPS, then with a
                # date, then apply the tie-break rule within the best tier.
                with_gps = [r for r in rows if r["lat"] is not None]
                if with_gps:
                    return _tiebreak(with_gps)
                with_date = [r for r in rows if r["taken_at"]]
                if with_date:
                    return _tiebreak(with_date)
                return _tiebreak(rows)

            removed = 0
            freed_bytes = 0
            errors = 0
            acting = delete or dedupe_index

            from rich.progress import Progress, BarColumn, MofNCompleteColumn, SpinnerColumn
            prog_ctx = Progress(
                SpinnerColumn(), BarColumn(), MofNCompleteColumn(),
                console=Console(stderr=True),
            ) if (acting and not verbose) else None

            def _run(track=None):
                nonlocal removed, freed_bytes, errors
                for i, h in enumerate(hashes):
                    if limit and i >= limit:
                        break
                    if track:
                        track()
                    rows = conn.execute(
                        f"""SELECT f.id, f.path, f.size, f.mtime,
                                   pm.lat AS lat, pm.taken_at AS taken_at
                            FROM files f LEFT JOIN photo_meta pm ON pm.file_id=f.id
                            WHERE f.{key_col} = ? AND f.missing = 0
                              AND f.dup_of IS NULL
                              {extra.replace('kind', 'f.kind')}""",
                        (h["gkey"],),
                    ).fetchall()
                    if len(rows) < 2:
                        continue
                    keeper = _picker(rows)
                    losers = [r for r in rows if r["id"] != keeper["id"]]

                    if verbose:
                        console.print(
                            f"\n[bold]{str(h['gkey'])[:16]}[/bold] "
                            f"({len(rows)} copies)"
                        )
                        console.print(f"  [green]keep[/green] {keeper['path']}")
                        for r in losers:
                            console.print(f"  [red]dup [/red] {r['path']}")

                    if not acting:
                        continue
                    for r in losers:
                        try:
                            if delete:
                                p = Path(r["path"])
                                if p.exists():
                                    p.unlink()
                                freed_bytes += r["size"] or 0
                                delete_file_row(conn, r["id"])
                            else:
                                # --dedupe-index: mark as a duplicate of the
                                # keeper. Non-destructive (row + file stay)
                                # and durable — rescan leaves dup_of alone,
                                # so it also works for visual dupes that
                                # hash-dedup can't re-skip.
                                conn.execute(
                                    "UPDATE files SET dup_of=? WHERE id=?",
                                    (keeper["id"], r["id"]),
                                )
                            removed += 1
                        except OSError as e:
                            errors += 1
                            console.print(f"    [red]error:[/red] {e}")
                    conn.commit()

            # Pure dry run (no --delete/--dedupe-index, not --verbose) has
            # nothing to do per group — the summary above is the whole
            # output. Skip the per-group loop so it returns immediately
            # instead of running 20k+ no-op queries (which looked hung).
            if acting or verbose:
                if prog_ctx:
                    with prog_ctx as prog:
                        task = prog.add_task("dedup", total=min(
                            total_groups, limit or total_groups))
                        _run(track=lambda: prog.advance(task))
                else:
                    _run()

            if delete:
                mb = freed_bytes / (1024 * 1024)
                console.print(
                    f"\n[green]removed {removed} files from disk[/green] "
                    f"(~{mb:.1f} MB), errors: {errors}"
                )
            elif dedupe_index:
                console.print(
                    f"\n[green]hid {removed} redundant copies[/green] "
                    f"(files + rows kept, marked dup_of), errors: {errors}"
                )
            else:
                console.print(
                    f"\n[dim]would resolve {total_dup_files} redundant files[/dim]"
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

    @parent.command(name="refresh")
    def cmd_refresh(
        workers: int = typer.Option(
            4, "--workers", "-w",
            help="Passed through to the underlying `index` run.",
        ),
        walk_workers: int | None = typer.Option(
            None, "--walk-workers",
            help="Passed through to the underlying `rescan` run.",
        ),
        lock: bool = typer.Option(
            True, "--lock/--no-lock",
            help="Refuse to start if another refresh is already running "
                 "(via an fcntl lock in the cache dir). On by default.",
        ),
    ) -> None:
        """Cron-friendly one-shot: rescan → babysit-index → cluster-faces.

        Fire this from cron / launchd on a schedule and the DB will keep
        current. All three phases append timestamped output to
        ``<cache>/logs/refresh.log`` for post-hoc inspection. Stdout is
        quiet on success and prints one summary line on failure, so
        cron's mail-on-nonzero-output alerts you only when something
        needs attention.

        Exit codes:
          0   every phase succeeded
          1   one or more phases returned non-zero
          2   another refresh was already running (lock held)

        Example crontab (every hour on the hour):

            0 * * * * cd /path/to/image-wizard && \\
                /Users/you/.local/bin/uv run image-wizard refresh

        Or as a launchd LaunchAgent — see the README section
        "Scheduled refresh".
        """
        import datetime as _dt
        import fcntl
        import subprocess
        import sys

        cfg = config.load()
        db.init(cfg.db_path)
        console = Console()
        log_dir = cfg.cache_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "refresh.log"
        lock_path = cfg.cache_dir / "refresh.lock"

        lock_fh = None
        if lock:
            try:
                lock_fh = open(lock_path, "w")
                fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_fh.write(str(os.getpid()))
                lock_fh.flush()
            except (OSError, BlockingIOError):
                # Another refresh is already running — that's fine, we
                # just exit with a distinct code so cron doesn't mail on
                # "expected concurrent skip".
                print(
                    f"another refresh is already running (lock at "
                    f"{lock_path})", file=sys.stderr,
                )
                raise typer.Exit(2)

        def _stamp() -> str:
            return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # All subprocesses funnel their stdout+stderr into the same log
        # file. We use python -m imagewizard.cli rather than the
        # `image-wizard` script so we avoid depending on PATH — cron's
        # environment is spartan.
        def phase(name: str, argv: list[str]) -> int:
            with open(log_path, "a") as log:
                log.write(f"\n[{_stamp()}] === {name} ===\n")
                log.flush()
                proc = subprocess.run(
                    [sys.executable, "-m", "imagewizard.cli", *argv],
                    stdout=log, stderr=subprocess.STDOUT,
                )
                log.write(
                    f"[{_stamp()}] === {name} exited "
                    f"{proc.returncode} ===\n"
                )
            return proc.returncode

        phases: list[tuple[str, list[str], int]] = []
        rescan_args = ["rescan"]
        if walk_workers is not None:
            rescan_args += ["--walk-workers", str(walk_workers)]
        phases.append(("rescan", rescan_args, phase("rescan", rescan_args)))

        # Passthrough --workers to `index` via babysit-index's `--` fence.
        babysit_args = ["babysit-index", "--", "--workers", str(workers)]
        phases.append(("babysit-index", babysit_args,
                       phase("babysit-index", babysit_args)))

        phases.append(("cluster-faces", ["cluster-faces"],
                       phase("cluster-faces", ["cluster-faces"])))

        # OCR (Apple Vision) — only if available on this box, so a
        # non-macOS / no-pyobjc host doesn't fail the refresh.
        from . import ocr as _ocr_mod
        if _ocr_mod.available():
            phases.append(("ocr", ["ocr"], phase("ocr", ["ocr"])))

        # Summary — silent on success, one line to stdout on failure so
        # cron mails it.
        failures = [name for name, _, rc in phases if rc != 0]
        with open(log_path, "a") as log:
            log.write(
                f"[{_stamp()}] refresh done "
                f"({len(phases) - len(failures)}/{len(phases)} phases OK)\n"
            )
        if failures:
            print(
                f"image-wizard refresh: {len(failures)} phase(s) failed: "
                f"{', '.join(failures)}. See {log_path}",
                file=sys.stderr,
            )
            raise typer.Exit(1)

        if lock_fh is not None:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
            lock_fh.close()

    @parent.command(
        name="babysit-index",
        context_settings={
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        },
    )
    def cmd_babysit(
        ctx: typer.Context,
        max_retries: int = typer.Option(
            30, "--max-retries",
            help="Stop after this many consecutive crashes (default: 30).",
        ),
        max_tombstones: int = typer.Option(
            100, "--max-tombstones",
            help="Stop after auto-tombstoning this many files (default: 100).",
        ),
        auto_tombstone: bool = typer.Option(
            True, "--auto-tombstone/--no-auto-tombstone",
            help="Mark the last in-flight file as decode_failed after a "
                 "crash so the next run skips it. On by default — that's "
                 "the whole point.",
        ),
    ) -> None:
        """Run ``index`` in a loop, auto-tombstoning crashers.

        Fire-and-forget wrapper for a flaky-ML-library environment. Runs
        the same ``index`` subprocess you'd run by hand; if it exits
        non-zero (or gets killed), reads the checkpoint log to find the
        file that was in flight at crash time, marks it ``decode_failed
        =1`` (reversible with ``clear-failures``), and restarts.

        Stops when:
          * ``index`` exits 0 (queue empty — normal completion),
          * ``--max-retries`` consecutive crashes,
          * ``--max-tombstones`` files marked (safety cap in case a
            whole *class* of files is broken and we shouldn't keep
            tombstoning them),
          * the same file id would be tombstoned twice in a row
            (means tombstoning didn't help — something else is wrong).

        Extra positional args are passed through to ``index``:

            uv run image-wizard babysit-index -- --workers 4 --no-clip
        """
        import subprocess
        import sys

        cfg = config.load()
        db.init(cfg.db_path)
        console = Console()
        log_path = cfg.cache_dir / "logs" / "index.log"

        # Everything after `babysit-index` and its own flags gets passed
        # through to `index` verbatim. Typer collects them in
        # ctx.args when the two context_settings above are set.
        passthrough_args = list(ctx.args)

        crashes = 0
        tombstones = 0
        last_tombstoned_id: int | None = None

        while True:
            cmd = [
                sys.executable, "-m", "imagewizard.cli", "index",
                *passthrough_args,
            ]
            console.rule(
                f"[bold]run #{crashes + tombstones + 1}[/bold]  "
                f"(crashes={crashes}, tombstoned={tombstones})"
            )
            console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
            result = subprocess.run(cmd)

            if result.returncode == 0:
                console.rule("[green]index completed cleanly[/green]")
                console.print(
                    f"total crashes handled: {crashes}, "
                    f"files auto-tombstoned: {tombstones}"
                )
                return

            console.print(
                f"[red]index exited with code {result.returncode}[/red]"
            )
            crashes += 1

            if not auto_tombstone:
                console.print(
                    "[yellow]--no-auto-tombstone: restarting without "
                    "marking the crasher.[/yellow]"
                )
                if crashes >= max_retries:
                    console.print(
                        f"[red]max retries ({max_retries}) reached[/red]"
                    )
                    return
                continue

            in_flight = _find_last_in_flight(log_path)
            if in_flight is None:
                console.print(
                    "[yellow]could not identify in-flight file from log — "
                    "restarting without tombstoning.[/yellow]"
                )
                if crashes >= max_retries:
                    console.print(
                        f"[red]max retries ({max_retries}) reached[/red]"
                    )
                    return
                continue

            file_id, path_hint = in_flight
            if file_id == last_tombstoned_id:
                console.print(
                    f"[red]file #{file_id} would be tombstoned twice in a "
                    f"row — aborting.[/red] Something else is wrong."
                )
                return

            conn = db.connect(cfg.db_path)
            try:
                conn.execute(
                    "UPDATE files SET decode_failed=1, decode_error=? "
                    "WHERE id=?",
                    (f"babysit auto-tombstone after crash #{crashes}"[:500],
                     file_id),
                )
            finally:
                conn.close()

            tombstones += 1
            last_tombstoned_id = file_id
            console.print(
                f"[yellow]tombstoned file #{file_id}[/yellow] {path_hint}"
            )

            if tombstones >= max_tombstones:
                console.print(
                    f"[red]max tombstones ({max_tombstones}) reached — "
                    "something is systemically wrong; investigate before "
                    "raising the cap.[/red]"
                )
                return
            crashes = 0  # progress made, reset the consecutive counter

    @parent.command(name="find-live-photos")
    def cmd_find_live_photos(
        rescan_all: bool = typer.Option(
            False, "--rescan-all",
            help="Re-check even files that already have live_photo_of "
                 "set. Use after fixing a bad match by hand.",
        ),
    ) -> None:
        """Detect iPhone Live Photo pairs and hide the .MOV companion.

        Live Photos land on disk as a HEIC + MOV pair with the same
        basename (``IMG_1234.HEIC`` + ``IMG_1234.MOV``). The .MOV is
        1–2s of motion around the shot — showing it separately in the
        timeline looks like a duplicate. This helper marks each such
        .MOV as ``live_photo_of=<the HEIC's id>``; the pipeline and
        the web UI then hide it. Reverse by clearing the column with
        SQL if you want the MOV back in the timeline.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            n = detect_live_photos(conn, only_new=not rescan_all)
            console.print(f"[green]flagged {n} Live Photo companion(s)[/green]")
        finally:
            conn.close()

    @parent.command(name="check-missing")
    def cmd_check_missing(
        dry_run: bool = typer.Option(
            False, "--dry-run",
            help="Report what would be marked missing without changing the DB.",
        ),
        limit: int = typer.Option(
            0, "--limit",
            help="Stop after checking this many files (0 = all).",
        ),
    ) -> None:
        """Mark files whose source path is no longer accessible as missing.

        Sweeps every ``missing=0`` row in ``files`` and stats each
        path. Anything that doesn't exist on disk (or on an unmounted
        network volume) gets ``missing=1`` so the web UI stops linking
        to it. Reverse via ``rescan`` — a subsequent rescan that finds
        the path again unsets the flag automatically.

        Cheap on local disk; slower on network mounts (one ``stat``
        per file). Runs single-threaded for predictable I/O.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            rows = conn.execute(
                "SELECT id, path FROM files WHERE missing=0"
            ).fetchall()
            if limit:
                rows = rows[:limit]
            console.print(f"[dim]checking {len(rows)} file(s)…[/dim]")
            gone: list[tuple[int, str]] = []
            for r in rows:
                if not Path(r["path"]).exists():
                    gone.append((r["id"], r["path"]))
            console.print(f"[bold]{len(gone)} missing on disk[/bold]")
            for fid, path in gone[:20]:
                console.print(f"  #{fid:7d}  {path}")
            if len(gone) > 20:
                console.print(f"  ... and {len(gone) - 20} more")
            if dry_run:
                console.print("[yellow]dry-run — nothing changed[/yellow]")
                return
            for fid, _ in gone:
                conn.execute(
                    "UPDATE files SET missing=1 WHERE id=?", (fid,)
                )
            console.print(
                f"[green]marked {len(gone)} row(s) as missing[/green]"
            )
        finally:
            conn.close()

    @parent.command(name="ocr")
    def cmd_ocr(
        workers: int = typer.Option(
            4, "--workers", "-w", help="Parallel OCR threads."
        ),
        limit: int = typer.Option(
            0, "--limit", help="Stop after this many files (0 = all)."
        ),
        redo: bool = typer.Option(
            False, "--redo", help="Re-OCR files already processed."
        ),
        path_glob: str = typer.Option(
            "", "--path", help="Only OCR files matching this LIKE pattern."
        ),
    ) -> None:
        """Recognize text in photos with Apple Vision, for full-text search.

        On-device OCR (no cloud) over each photo's cached thumbnail —
        street signs, storefronts, book covers, whiteboards, screenshots.
        The text goes into an FTS5 index; search it from the web UI's
        "text in photo" box. Incremental: only un-OCR'd files are done
        unless --redo. macOS only (needs the Vision framework).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .thumbs import thumb_path
        from . import ocr as ocr_mod

        console = Console()
        if not ocr_mod.available():
            console.print(
                "[red]Apple Vision OCR unavailable[/red] — needs macOS + "
                "`uv pip install pyobjc-framework-Vision pyobjc-framework-Quartz`."
            )
            raise typer.Exit(1)

        cfg = config.load()
        db.init(cfg.db_path)
        conn = db.connect(cfg.db_path)
        try:
            where = ["missing=0", "too_small=0", "decode_failed=0",
                     "dup_of IS NULL", "content_hash != ''"]
            params: list = []
            if not redo:
                where.append("ocr_done=0")
            if path_glob:
                where.append("path LIKE ?")
                params.append(path_glob)
            rows = conn.execute(
                f"SELECT id, content_hash FROM files WHERE {' AND '.join(where)}",
                params,
            ).fetchall()
            if limit:
                rows = rows[:limit]
            if not rows:
                console.print("[green]nothing to OCR[/green]")
                return
            console.print(f"[dim]OCR over {len(rows)} photo(s)…[/dim]")

            def ocr_one(r):
                tp = thumb_path(cfg.cache_dir, r["content_hash"])
                if not tp.exists():
                    return (r["id"], None)
                return (r["id"], ocr_mod.recognize_text(tp))

            done = with_text = 0
            pending: list[tuple[int, str]] = []
            from rich.progress import (
                Progress, BarColumn, MofNCompleteColumn, SpinnerColumn,
                TimeRemainingColumn,
            )
            with ThreadPoolExecutor(max_workers=workers) as pool, \
                    Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(),
                             TimeRemainingColumn(), console=Console(stderr=True)) as prog:
                task = prog.add_task("ocr", total=len(rows))
                futs = {pool.submit(ocr_one, r): r for r in rows}
                for fut in as_completed(futs):
                    fid, text = fut.result()
                    if text is not None:
                        pending.append((fid, text))
                        if text.strip():
                            with_text += 1
                    done += 1
                    prog.advance(task)
                    if len(pending) >= 200:
                        _flush_ocr(conn, pending)
                        pending = []
            if pending:
                _flush_ocr(conn, pending)
            console.print(
                f"[green]OCR'd {done} photos[/green]; {with_text} contained text."
            )
        finally:
            conn.close()

    @parent.command(name="check-readable")
    def cmd_check_readable(
        videos_only: bool = typer.Option(
            False, "--videos-only", help="Only check video files."
        ),
        images_only: bool = typer.Option(
            False, "--images-only", help="Only check image files."
        ),
        path_glob: str = typer.Option(
            "", "--path",
            help="Only check files whose path matches this LIKE pattern "
                 "(e.g. '%/AAA-CLEANMEUP/%').",
        ),
        workers: int = typer.Option(
            8, "--workers", "-w", help="Parallel decode threads."
        ),
        dry_run: bool = typer.Option(
            False, "--dry-run",
            help="Report broken files without tombstoning them.",
        ),
        limit: int = typer.Option(
            0, "--limit", help="Stop after checking this many files (0 = all)."
        ),
    ) -> None:
        """Find files that can't be decoded and tombstone them.

        Attempts to decode every live, not-already-failed file (an image
        via the normal decode path, a video via its poster-frame
        extractor). Anything that throws — truncated JPEGs, corrupt
        ``.MOV`` files with no moov atom, unsupported RAW variants — is
        marked ``decode_failed=1`` so the index pipeline and the
        duplicate scanner skip it instead of erroring on it every run.

        Reverse with ``clear-failures``. Native decoder noise (ffmpeg /
        libheif chatter) is suppressed so only the summary prints.

        Scope it with ``--videos-only`` / ``--images-only`` / ``--path``
        to check just a suspect subset quickly, rather than re-decoding
        the whole library.
        """
        import contextlib
        import sys as _sys
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            where = ["missing=0", "decode_failed=0", "too_small=0",
                     "content_hash != ''"]
            params: list = []
            if videos_only:
                where.append("kind='video'")
            if images_only:
                where.append("COALESCE(kind,'image')!='video'")
            if path_glob:
                where.append("path LIKE ?")
                params.append(path_glob)
            rows = conn.execute(
                f"SELECT id, path FROM files WHERE {' AND '.join(where)}",
                params,
            ).fetchall()
            if limit:
                rows = rows[:limit]
            if not rows:
                console.print("[green]nothing to check[/green]")
                return
            console.print(f"[dim]checking {len(rows)} file(s)…[/dim]")

            def check_one(r):
                p = Path(r["path"])
                if not p.exists():
                    return (r["id"], "missing", "file not on disk")
                try:
                    if p.suffix.lower() in VIDEO_EXTS:
                        from .video import extract_poster
                        extract_poster(p)
                    else:
                        from .decode import load_image
                        load_image(p)
                    return (r["id"], "ok", None)
                except Exception as e:
                    return (r["id"], "fail", str(e).replace("\n", " ")[:300])

            # Redirect the process's stderr fd to /dev/null for the whole
            # batch so native ffmpeg / libheif / OpenCV chatter (which
            # writes straight to fd 2, bypassing Python) doesn't flood the
            # terminal. The progress bar goes to stdout so it stays
            # visible. Exceptions are still captured in Python, so we
            # keep the real failure reason.
            @contextlib.contextmanager
            def _quiet_native_stderr():
                devnull = os.open(os.devnull, os.O_WRONLY)
                saved = os.dup(2)
                _sys.stderr.flush()
                os.dup2(devnull, 2)
                try:
                    yield
                finally:
                    _sys.stderr.flush()
                    os.dup2(saved, 2)
                    os.close(devnull)
                    os.close(saved)

            broken: list[tuple[int, str, str]] = []
            missing: list[int] = []
            from rich.progress import (
                Progress, BarColumn, MofNCompleteColumn, SpinnerColumn,
                TimeRemainingColumn,
            )
            with _quiet_native_stderr(), \
                    ThreadPoolExecutor(max_workers=workers) as pool, \
                    Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(),
                             TimeRemainingColumn(), console=Console()) as prog:
                task = prog.add_task("checking", total=len(rows))
                futs = {pool.submit(check_one, r): r for r in rows}
                for fut in as_completed(futs):
                    fid, status, reason = fut.result()
                    if status == "fail":
                        broken.append((fid, futs[fut]["path"], reason))
                    elif status == "missing":
                        missing.append(fid)
                    prog.advance(task)

            console.print(
                f"[bold]{len(broken)} unreadable[/bold], "
                f"{len(missing)} missing-on-disk, "
                f"{len(rows) - len(broken) - len(missing)} ok"
            )
            for fid, path, reason in broken[:20]:
                console.print(f"  [red]#{fid}[/red] {path}")
                console.print(f"       {reason}")
            if len(broken) > 20:
                console.print(f"  ... and {len(broken) - 20} more")

            if dry_run:
                console.print("[yellow]dry-run — nothing changed[/yellow]")
                return
            for fid, _p, reason in broken:
                conn.execute(
                    "UPDATE files SET decode_failed=1, decode_error=? WHERE id=?",
                    (f"check-readable: {reason}"[:500], fid),
                )
            # Files that vanished from disk mid-check → mark missing.
            for fid in missing:
                conn.execute("UPDATE files SET missing=1 WHERE id=?", (fid,))
            console.print(
                f"[green]tombstoned {len(broken)} unreadable file(s)[/green]"
                + (f", marked {len(missing)} missing" if missing else "")
                + ". Reverse unreadable ones with `clear-failures`."
            )
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

            # ---- Duplicate status -------------------------------------
            # Why is this photo showing (or not) as a duplicate? Surface
            # the byte hash, the perceptual hash, the durable dup_of flag,
            # and any siblings — exact-byte, exact-phash, and phash-near
            # (Hamming ≤ 8, which exact grouping in `find-duplicates`
            # misses). A sibling tagged VISIBLE is a live duplicate.
            try:
                phash = row["phash"]
                dup_of = row["dup_of"]
            except (IndexError, KeyError):
                phash = dup_of = None
            console.print("\n[bold]Duplicate status[/bold]")
            console.print(f"  phash: {phash or '—'}")
            if dup_of:
                dp = conn.execute(
                    "SELECT path FROM files WHERE id=?", (dup_of,)
                ).fetchone()
                console.print(
                    f"  [yellow]hidden as dup_of #{dup_of}[/yellow]  "
                    f"{dp['path'] if dp else '(keeper missing)'}"
                )
            else:
                console.print("  dup_of: — (visible in the timeline)")

            def _tag(r):
                return (f" dup_of#{r['dup_of']}" if r["dup_of"]
                        else " [red]VISIBLE[/red]")

            same_bytes = conn.execute(
                """SELECT id, path, dup_of FROM files
                   WHERE content_hash=? AND id!=? AND missing=0""",
                (row["content_hash"], fid),
            ).fetchall()
            if same_bytes:
                console.print(
                    f"  [cyan]{len(same_bytes)} byte-identical sibling(s)[/cyan] "
                    "(same content_hash — default `find-duplicates` catches these):"
                )
                for s in same_bytes[:8]:
                    console.print(f"    #{s['id']}{_tag(s)}  {s['path']}")

            if phash:
                try:
                    target = int(phash, 16)
                    others = conn.execute(
                        """SELECT id, path, phash, dup_of FROM files
                           WHERE phash IS NOT NULL AND phash!='' AND id!=?
                             AND missing=0""",
                        (fid,),
                    ).fetchall()
                    near = []
                    for r2 in others:
                        try:
                            d = bin(target ^ int(r2["phash"], 16)).count("1")
                        except ValueError:
                            continue
                        if d <= 8:
                            near.append((d, r2))
                    near.sort(key=lambda t: t[0])
                    if near:
                        console.print(
                            f"  [cyan]{len(near)} visually-near sibling(s)[/cyan] "
                            "(phash Hamming ≤ 8; dist 0 = `--visual` catches it, "
                            "dist > 0 needs near-dup matching):"
                        )
                        for d, r2 in near[:8]:
                            kind = "exact" if d == 0 else f"dist {d}"
                            console.print(
                                f"    #{r2['id']} ({kind}){_tag(r2)}  {r2['path']}"
                            )
                except ValueError:
                    pass

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

    @parent.command(name="compare-roots")
    def cmd_compare_roots(
        prefix_a: str = typer.Argument(
            ..., help="Path substring for the first tree, e.g. 'Photos export'."),
        prefix_b: str = typer.Argument(
            ..., help="Path substring for the second tree, e.g. 'photoslibrary'."),
        key: str = typer.Option(
            "exif", "--key",
            help="Same-photo key: 'exif' (taken_at+filename — survives "
                 "re-encoding, the default), 'phash' (perceptual; needs "
                 "find-duplicates --visual first), 'taken' (timestamp only), "
                 "or 'name' (filename only)."),
        examples: int = typer.Option(
            0, "--examples", "-e",
            help="Print up to N example photos unique to each side."),
        exclude: str = typer.Option(
            "", "--exclude",
            help="Drop paths containing this substring from both sides "
                 "before comparing, e.g. 'Previews' to skip Apple Photos "
                 "preview derivatives inside a .photoslibrary package."),
    ) -> None:
        """Compare two scan trees: counts, overlap, and a superset verdict.

        The same library re-exported into two folder layouts has *different*
        bytes per photo (so content_hash won't match across them). This keys
        on capture time + filename instead — both preserved through
        re-encoding — to answer "which tree is bigger, and is one a superset
        of the other?" before you drop a redundant scan root.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            def load(prefix: str):
                sql = ("""SELECT f.id, f.path, f.phash, pm.taken_at
                          FROM files f LEFT JOIN photo_meta pm ON pm.file_id=f.id
                          WHERE f.missing=0 AND f.kind='image' AND f.path LIKE ?""")
                params = [f"%{prefix}%"]
                if exclude:
                    sql += " AND f.path NOT LIKE ?"
                    params.append(f"%{exclude}%")
                return conn.execute(sql, params).fetchall()

            def keyf(r):
                if key == "phash":
                    return r["phash"] or None
                if key == "name":
                    return os.path.basename(r["path"]).lower()
                if key == "taken":
                    return r["taken_at"] or None
                t = r["taken_at"]                       # 'exif' (default)
                return f"{t}|{os.path.basename(r['path']).lower()}" if t else None

            def keyset(rows):
                keyed, unkeyed = {}, 0
                for r in rows:
                    k = keyf(r)
                    if k is None:
                        unkeyed += 1
                    else:
                        keyed.setdefault(k, r)
                return keyed, unkeyed

            rows_a, rows_b = load(prefix_a), load(prefix_b)
            ka, ua = keyset(rows_a)
            kb, ub = keyset(rows_b)
            sa, sb = set(ka), set(kb)
            only_a, only_b, both = sa - sb, sb - sa, sa & sb

            def line(lbl, rows, keyed, unk):
                extra = f", {unk} unkeyable (no date)" if unk else ""
                console.print(f"[bold]{lbl}[/bold]: {len(rows)} photos, "
                              f"{len(keyed)} distinct{extra}")
            line(f"A ~{prefix_a!r}", rows_a, sa, ua)
            line(f"B ~{prefix_b!r}", rows_b, sb, ub)
            console.print(f"\n[dim]same-photo key = {key}[/dim]")
            console.print(f"  in both:  {len(both)}")
            console.print(f"  A only:   {len(only_a)}")
            console.print(f"  B only:   {len(only_b)}")

            console.print()
            if not sa and not sb:
                console.print("[yellow]nothing keyed[/yellow] — most photos "
                              "lack a capture date; retry with --key name")
            elif not only_a and not only_b:
                console.print("[green]identical sets[/green] — either root can go.")
            elif not only_a:
                console.print(f"[green]A ⊆ B[/green] — B is a superset. Every "
                              f"photo in A is also in B; dropping root A "
                              f"({len(rows_a)} files) loses nothing.")
            elif not only_b:
                console.print(f"[green]B ⊆ A[/green] — A is a superset. Every "
                              f"photo in B is also in A; dropping root B "
                              f"({len(rows_b)} files) loses nothing.")
            else:
                console.print(f"[yellow]partial overlap[/yellow] — A has "
                              f"{len(only_a)} photos not in B, B has "
                              f"{len(only_b)} not in A. Neither is a clean "
                              f"superset; dedupe rather than drop a root.")
            if (ua or ub) and (only_a or only_b):
                console.print("[dim](photos with no capture date aren't keyed; "
                              "they're excluded from the verdict.)[/dim]")

            if examples:
                def show(lbl, keys, kmap):
                    if not keys:
                        return
                    console.print(f"\n[bold]{lbl} (up to {examples}):[/bold]")
                    for k in list(keys)[:examples]:
                        console.print(f"  {kmap[k]['path']}")
                show("A only", only_a, ka)
                show("B only", only_b, kb)
        finally:
            conn.close()

    @parent.command(name="prune-path")
    def cmd_prune_path(
        substring: str = typer.Argument(
            ..., help="Delete indexed rows whose path contains this substring."),
        delete: bool = typer.Option(
            False, "--delete", help="Actually delete (default is a dry run)."),
        show: int = typer.Option(
            20, "--show", help="How many example paths to print."),
    ) -> None:
        """Delete indexed rows whose path contains a substring (dry-run default).

        For pruning derivative junk that should never have been indexed
        (e.g. Apple Photos `Previews/` inside a .photoslibrary) or dropping
        a scan tree you've decided to abandon. Cascades through detections /
        faces / vec_* / ocr_fts via delete_file_row. The files on disk are
        NOT touched — only the index rows. Dry-run by default; pass --delete
        to act.

        Note: this removes rows but leaves scan_roots alone, so a future
        scan of a still-listed root could re-add matching files. For
        derivative dirs, the walker already skips them (Previews/Thumbnails/
        resources inside a photo library) so they won't come back; to drop
        a whole root, also remove it from the scan set.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            rows = conn.execute(
                "SELECT id, path FROM files WHERE path LIKE ?",
                (f"%{substring}%",),
            ).fetchall()
            console.print(
                f"[bold]{len(rows)} row(s)[/bold] match path substring "
                f"{substring!r}"
            )
            for r in rows[:show]:
                console.print(f"  #{r['id']:8d}  {r['path']}")
            if len(rows) > show:
                console.print(f"  ... and {len(rows) - show} more")
            if not delete:
                console.print(
                    "[yellow]dry run[/yellow] — pass --delete to remove "
                    "these rows (files on disk are never touched)"
                )
                return
            if not rows:
                return

            from rich.progress import Progress, BarColumn, MofNCompleteColumn, SpinnerColumn
            n = 0
            with Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(),
                          console=Console(stderr=True)) as prog:
                task = prog.add_task("pruning", total=len(rows))
                for r in rows:
                    delete_file_row(conn, r["id"])
                    n += 1
                    if n % 500 == 0:
                        conn.commit()
                    prog.advance(task)
            conn.commit()
            console.print(
                f"[green]deleted {n} row(s)[/green] "
                "(source files on disk untouched)"
            )
        finally:
            conn.close()

    @parent.command(name="backfill-dates")
    def cmd_backfill_dates(
        apply: bool = typer.Option(
            False, "--apply",
            help="Write the inferred dates (default is a dry run)."),
        redo: bool = typer.Option(
            False, "--redo",
            help="Also re-infer dates a previous backfill set "
                 "(date_inferred=1), e.g. after improving the patterns."),
        path: str = typer.Option(
            "", "--path",
            help="Only consider files whose path contains this substring."),
        show: int = typer.Option(
            15, "--show", help="How many example rows to print."),
    ) -> None:
        """Infer taken_at from folder/filename dates where EXIF has none.

        Old scans and early-digital photos often carry no EXIF capture
        date, so they sit undated (mis-sorted) in the timeline — even
        though the library filed them under `2004/10/25/` or an event
        folder like "August 4, 2005", or the camera stamped the date into
        the filename (`IMG_20040825_143000.jpg`). This recovers that date
        into `photo_meta.taken_at` and marks it `date_inferred=1` so it's
        distinguishable from a real EXIF date and never overwrites one.

        Dry-run by default; pass --apply to write.
        """
        from . import datefind
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            # meta_done=1: only files that already went through EXIF
            # extraction and genuinely have no date — never pre-empt a file
            # whose metadata stage just hasn't run yet (it'll get a real date).
            where = ["f.missing = 0", "f.meta_done = 1"]
            params: list = []
            if redo:
                where.append("(pm.taken_at IS NULL OR pm.date_inferred = 1)")
            else:
                where.append("pm.taken_at IS NULL")
            if path:
                where.append("f.path LIKE ?")
                params.append(f"%{path}%")
            rows = conn.execute(
                f"""SELECT f.id, f.path FROM files f
                    LEFT JOIN photo_meta pm ON pm.file_id = f.id
                    WHERE {' AND '.join(where)}""",
                params,
            ).fetchall()

            inferred: list[tuple[int, str]] = []
            misses = 0
            from collections import Counter
            by_year: Counter = Counter()
            for r in rows:
                d = datefind.infer_date(r["path"])
                if d:
                    inferred.append((r["id"], d))
                    by_year[d[:4]] += 1
                else:
                    misses += 1

            console.print(
                f"[bold]{len(rows)} undated file(s)[/bold]; "
                f"[green]{len(inferred)} inferable[/green], "
                f"[yellow]{misses} not[/yellow]"
            )
            if by_year:
                span = ", ".join(f"{y}:{n}" for y, n in sorted(by_year.items()))
                console.print(f"  by inferred year → {span}")
            for fid, d in inferred[:show]:
                p = next(r["path"] for r in rows if r["id"] == fid)
                console.print(f"  [cyan]{d[:10]}[/cyan]  {p}")
            if len(inferred) > show:
                console.print(f"  ... and {len(inferred) - show} more")

            if not apply:
                console.print("[yellow]dry run[/yellow] — pass --apply to write "
                              "these into photo_meta.taken_at")
                return
            if not inferred:
                return

            from rich.progress import Progress, BarColumn, MofNCompleteColumn, SpinnerColumn
            n = 0
            with Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(),
                          console=Console(stderr=True)) as prog:
                task = prog.add_task("writing", total=len(inferred))
                for fid, d in inferred:
                    # Upsert: many files already have a photo_meta row (with
                    # NULL taken_at); some have none. Never touch a real date
                    # — the candidate set is taken_at IS NULL (or --redo's own
                    # prior guesses).
                    conn.execute(
                        """INSERT INTO photo_meta (file_id, taken_at, date_inferred)
                           VALUES (?, ?, 1)
                           ON CONFLICT(file_id) DO UPDATE SET
                             taken_at = excluded.taken_at, date_inferred = 1""",
                        (fid, d),
                    )
                    conn.execute("UPDATE files SET meta_done = 1 WHERE id = ?", (fid,))
                    n += 1
                    if n % 500 == 0:
                        conn.commit()
                    prog.advance(task)
            conn.commit()
            console.print(f"[green]backfilled {n} date(s)[/green] "
                          "(marked date_inferred=1)")
        finally:
            conn.close()
