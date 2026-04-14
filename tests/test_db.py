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
