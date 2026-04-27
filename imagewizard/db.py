"""SQLite schema, connection helpers, and sqlite-vec loading.

This module is the single source of truth for the on-disk schema. All other
modules read and write through connections obtained here.

Design notes:

* `files` is the canonical record of a physical file on disk. It carries the
  content hash used for deduplication and the mtime/size pair used for fast
  incremental scans.
* `photo_meta` is one-to-one with `files` and holds everything extracted from
  EXIF plus reverse-geocoded place names.
* `detections` and `faces` are normalized (many-per-file) so label/cluster
  queries are cheap SQL.
* `vec_clip` and `vec_faces` are sqlite-vec virtual tables. Their `rowid`
  matches `files.id` and `faces.id` respectively, so joining vectors back to
  metadata is a plain `JOIN ... USING (rowid)`.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

SCHEMA_VERSION = 5

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    size         INTEGER NOT NULL,
    mtime        REAL NOT NULL,
    mime         TEXT,
    width        INTEGER,
    height       INTEGER,
    indexed_at   REAL NOT NULL,
    -- ML stage flags; set as each stage completes so re-runs are cheap.
    meta_done    INTEGER NOT NULL DEFAULT 0,
    yolo_done    INTEGER NOT NULL DEFAULT 0,
    faces_done   INTEGER NOT NULL DEFAULT 0,
    clip_done    INTEGER NOT NULL DEFAULT 0,
    missing      INTEGER NOT NULL DEFAULT 0  -- tombstone for removed files
);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_missing ON files(missing);

CREATE TABLE IF NOT EXISTS photo_meta (
    file_id      INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
    taken_at     TEXT,
    camera_make  TEXT,
    camera_model TEXT,
    lens         TEXT,
    iso          INTEGER,
    aperture     REAL,
    shutter      TEXT,
    focal_mm     REAL,
    lat          REAL,
    lon          REAL,
    alt          REAL,
    city         TEXT,
    region       TEXT,
    country      TEXT
);
CREATE INDEX IF NOT EXISTS idx_meta_taken ON photo_meta(taken_at);
CREATE INDEX IF NOT EXISTS idx_meta_model ON photo_meta(camera_model);
CREATE INDEX IF NOT EXISTS idx_meta_geo   ON photo_meta(lat, lon);

CREATE TABLE IF NOT EXISTS detections (
    id      INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    label   TEXT NOT NULL,
    conf    REAL NOT NULL,
    x       REAL, y REAL, w REAL, h REAL
);
CREATE INDEX IF NOT EXISTS idx_det_file  ON detections(file_id);
CREATE INDEX IF NOT EXISTS idx_det_label ON detections(label);

CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    x           REAL, y REAL, w REAL, h REAL,
    det_score   REAL,
    cluster_id  INTEGER,
    person_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_faces_file    ON faces(file_id);
CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id);

-- Face cluster centroids (for stable IDs across re-runs).
CREATE TABLE IF NOT EXISTS face_clusters (
    cluster_id  INTEGER PRIMARY KEY,
    centroid    BLOB NOT NULL,   -- float32 * 512
    person_name TEXT,
    face_count  INTEGER NOT NULL DEFAULT 0
);

-- Virtual tables for sqlite-vec. Keyed by rowid = files.id / faces.id.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_clip  USING vec0(embedding float[512]);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_faces USING vec0(embedding float[512]);

-- Directories that have been passed to `scan`. Used by the About page.
CREATE TABLE IF NOT EXISTS scan_roots (
    path            TEXT PRIMARY KEY,
    last_scanned_at REAL NOT NULL
);

-- Small key/value table for global timestamps and misc app state
-- (last_index_at, last_cluster_at, ...). Avoids dedicated one-row tables.
CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Person identity (one per real-world person). Unlike `face_clusters`
-- which is an HDBSCAN artefact, a person is an editable identity that
-- can span multiple clusters and carry multiple names over time
-- (married name change, nickname, etc.).
CREATE TABLE IF NOT EXISTS persons (
    id           INTEGER PRIMARY KEY,
    primary_name TEXT NOT NULL,
    notes        TEXT,
    created_at   REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- Name epochs: a person was called <name> from start_date to end_date.
-- Either bound may be NULL (open-ended). When a face's photo date falls
-- inside an epoch, that's the name shown for the face.
CREATE TABLE IF NOT EXISTS person_names (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    start_date  TEXT,    -- "YYYY-MM-DD" or NULL = open
    end_date    TEXT,    -- "YYYY-MM-DD" or NULL = ongoing
    is_nickname INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pn_person ON person_names(person_id);
CREATE INDEX IF NOT EXISTS idx_pn_name   ON person_names(name COLLATE NOCASE);

-- Per-video frame index. V1 video support treated each video as a single
-- still (poster frame) and stored its detections / faces on the file row
-- with frame_id IS NULL. V2 samples multiple frames and stores one row
-- per sampled frame here; per-frame detections / faces / CLIP vectors
-- reference that frame_id so we can show "Alice at 0:23 in beach.mov".
CREATE TABLE IF NOT EXISTS frames (
    id      INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ts_sec  REAL NOT NULL,    -- seconds into the video
    width   INTEGER,
    height  INTEGER,
    UNIQUE(file_id, ts_sec)
);
CREATE INDEX IF NOT EXISTS idx_frames_file ON frames(file_id);

-- CLIP embeddings for individual video frames. Kept separate from
-- vec_clip (which is keyed by files.id) so the rowid namespaces don't
-- collide. Search across photos + videos is a UNION ALL query.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_clip_frames USING vec0(embedding float[512]);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a sqlite connection with sqlite-vec loaded and pragmas set."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit; use tx() for txns
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init(db_path: Path) -> None:
    """Create schema if missing, run additive migrations, and backfill."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        _migrate(conn)
        _backfill_persons(conn)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version(version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent additive migrations (only ADD COLUMN, never drop)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(face_clusters)")}
    if "person_id" not in cols:
        conn.execute(
            "ALTER TABLE face_clusters ADD COLUMN person_id INTEGER"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fc_person ON face_clusters(person_id)"
        )

    # decode_failed: tombstone for files that errored during decode (corrupt
    # JPEG, unsupported RAW variant, ...). Marking them avoids retrying the
    # decode every `index` run, which is wasteful especially over network
    # mounts. Cleared by `image-wizard clear-failures`.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(files)")}
    if "decode_failed" not in cols:
        conn.execute(
            "ALTER TABLE files ADD COLUMN decode_failed INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_decode_failed "
            "ON files(decode_failed)"
        )
    if "decode_error" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN decode_error TEXT")

    # too_small: tombstone for files that fail the dimension filter (icon
    # cache, Synology auto-generated thumbnails, screenshots of dialog
    # boxes...). Recording them once means the next scan recognises the
    # path immediately and skips the expensive PIL header read.
    if "too_small" not in cols:
        conn.execute(
            "ALTER TABLE files ADD COLUMN too_small INTEGER NOT NULL DEFAULT 0"
        )

    # kind / duration_sec: video support (V1). kind='image' is the legacy
    # default for any pre-existing row; new scans set it explicitly.
    # duration_sec is NULL for images, populated for videos at index time
    # by the poster-frame extractor.
    if "kind" not in cols:
        conn.execute(
            "ALTER TABLE files ADD COLUMN kind TEXT NOT NULL DEFAULT 'image'"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_kind ON files(kind)"
        )
    if "duration_sec" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN duration_sec REAL")

    # frame_id on detections / faces (V2 video support). NULL means
    # "applies to the whole file" (legacy photos and V1 video poster
    # frames). Foreign-key cascade is set in the table definition for
    # NEW deployments; ALTER TABLE in SQLite can't add FK so existing
    # databases get the column without a constraint — application code
    # cleans up frame deletions explicitly.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(detections)")}
    if "frame_id" not in cols:
        conn.execute("ALTER TABLE detections ADD COLUMN frame_id INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id)"
        )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(faces)")}
    if "frame_id" not in cols:
        conn.execute("ALTER TABLE faces ADD COLUMN frame_id INTEGER")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_faces_frame ON faces(frame_id)"
        )


def _backfill_persons(conn: sqlite3.Connection) -> None:
    """For each existing distinct person_name on face_clusters, create one
    persons row + one open-ended person_names row, and link the cluster.

    Idempotent: skips clusters that already have a person_id set, and
    reuses an existing person if one with the same primary_name exists.
    """
    rows = conn.execute(
        """SELECT cluster_id, person_name FROM face_clusters
           WHERE person_name IS NOT NULL
             AND person_name != ''
             AND person_id IS NULL"""
    ).fetchall()
    if not rows:
        return

    for r in rows:
        cid, name = r[0], r[1]
        # Reuse a person with this name if one already exists
        existing = conn.execute(
            "SELECT id FROM persons WHERE primary_name = ? COLLATE NOCASE LIMIT 1",
            (name,),
        ).fetchone()
        if existing:
            pid = existing[0]
        else:
            cur = conn.execute(
                "INSERT INTO persons (primary_name) VALUES (?)", (name,)
            )
            pid = cur.lastrowid
            conn.execute(
                "INSERT INTO person_names (person_id, name) VALUES (?, ?)",
                (pid, name),
            )
        conn.execute(
            "UPDATE face_clusters SET person_id = ? WHERE cluster_id = ?",
            (pid, cid),
        )


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert a key/value pair in app_meta."""
    conn.execute(
        "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Context manager for an explicit transaction on an autocommit conn."""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
