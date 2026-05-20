"""M0 smoke test: schema initializes and vector tables work."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from imagewizard import config, db


@pytest.fixture()
def tmp_cfg(tmp_path, monkeypatch) -> config.Config:
    monkeypatch.setenv("IMAGEWIZARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGEWIZARD_CACHE_DIR", str(tmp_path / "cache"))
    return config.load()


def test_init_creates_schema(tmp_cfg):
    db.init(tmp_cfg.db_path)
    assert tmp_cfg.db_path.exists()

    conn = db.connect(tmp_cfg.db_path)
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual_table')"
            )
        }
        for expected in (
            "files",
            "photo_meta",
            "detections",
            "faces",
            "face_clusters",
            "vec_clip",
            "vec_faces",
        ):
            assert expected in tables, f"missing table {expected}"

        v = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert v == db.SCHEMA_VERSION
    finally:
        conn.close()


def test_vec_roundtrip(tmp_cfg):
    """Confirm sqlite-vec extension is loaded and usable."""
    db.init(tmp_cfg.db_path)
    conn = db.connect(tmp_cfg.db_path)
    try:
        vec = struct.pack("512f", *([0.1] * 512))
        conn.execute("INSERT INTO vec_clip(rowid, embedding) VALUES (1, ?)", (vec,))
        row = conn.execute(
            "SELECT rowid FROM vec_clip WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
            (vec,),
        ).fetchone()
        assert row["rowid"] == 1
    finally:
        conn.close()


def test_idempotent_init(tmp_cfg):
    db.init(tmp_cfg.db_path)
    db.init(tmp_cfg.db_path)  # second call must not raise


def _insert_file(conn, path: str, file_id: int = 1) -> int:
    conn.execute(
        "INSERT INTO files (id, path, content_hash, size, mtime, indexed_at) "
        "VALUES (?, ?, 'h', 0, 0, 0)",
        (file_id, path),
    )
    return file_id


def _insert_vec(conn, table: str, rowid: int) -> None:
    vec = struct.pack("512f", *([0.1] * 512))
    conn.execute(f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)", (rowid, vec))


def test_cascade_clears_vec_clip(tmp_cfg):
    """Deleting a files row clears its vec_clip row via trigger."""
    db.init(tmp_cfg.db_path)
    conn = db.connect(tmp_cfg.db_path)
    try:
        fid = _insert_file(conn, "/a.jpg")
        _insert_vec(conn, "vec_clip", fid)
        conn.execute("DELETE FROM files WHERE id=?", (fid,))
        n = conn.execute("SELECT COUNT(*) FROM vec_clip WHERE rowid=?", (fid,)).fetchone()[0]
        assert n == 0
    finally:
        conn.close()


def test_cascade_clears_vec_faces_through_files(tmp_cfg):
    """Files delete -> faces cascade -> vec_faces cleared by trigger."""
    db.init(tmp_cfg.db_path)
    conn = db.connect(tmp_cfg.db_path)
    try:
        fid = _insert_file(conn, "/b.jpg")
        cur = conn.execute(
            "INSERT INTO faces (file_id, det_score) VALUES (?, 0.9)", (fid,)
        )
        face_id = cur.lastrowid
        _insert_vec(conn, "vec_faces", face_id)
        conn.execute("DELETE FROM files WHERE id=?", (fid,))
        n = conn.execute(
            "SELECT COUNT(*) FROM vec_faces WHERE rowid=?", (face_id,)
        ).fetchone()[0]
        assert n == 0
    finally:
        conn.close()


def test_cascade_clears_vec_clip_frames_through_files(tmp_cfg):
    """Files delete -> frames cascade -> vec_clip_frames cleared by trigger."""
    db.init(tmp_cfg.db_path)
    conn = db.connect(tmp_cfg.db_path)
    try:
        fid = _insert_file(conn, "/c.mov")
        cur = conn.execute(
            "INSERT INTO frames (file_id, ts_sec) VALUES (?, 1.0)", (fid,)
        )
        frame_id = cur.lastrowid
        _insert_vec(conn, "vec_clip_frames", frame_id)
        conn.execute("DELETE FROM files WHERE id=?", (fid,))
        n = conn.execute(
            "SELECT COUNT(*) FROM vec_clip_frames WHERE rowid=?", (frame_id,)
        ).fetchone()[0]
        assert n == 0
    finally:
        conn.close()
