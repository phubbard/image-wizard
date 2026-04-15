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

SCHEMA_VERSION = 1

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
    """Create schema if missing and stamp the version."""
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version(version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
    finally:
        conn.close()


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
