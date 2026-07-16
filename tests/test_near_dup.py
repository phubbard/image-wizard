"""find-duplicates --near: filename + phash-Hamming grouping."""
from __future__ import annotations


def _add(conn, path, phash):
    conn.execute(
        """INSERT INTO files(path, content_hash, size, mtime, indexed_at,
                             missing, kind, phash, too_small)
           VALUES(?,?,?,?,?,0,'image',?,0)""",
        (path, path.replace("/", "_"), 1, 0.0, 0.0, phash),
    )


def test_near_dup_needs_same_name_and_close_phash(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGEWIZARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGEWIZARD_CACHE_DIR", str(tmp_path / "cache"))
    from imagewizard import config, db
    cfg = config.load()
    db.init(cfg.db_path)
    conn = db.connect(cfg.db_path)
    # Same filename in two trees, phash 2 bits apart → SHOULD group.
    _add(conn, "/A/IMG_1.JPG", "0000000000000000")
    _add(conn, "/B/IMG_1.JPG", "0000000000000003")
    # Same filename, wildly different phash → must NOT join that group.
    _add(conn, "/C/IMG_1.JPG", "ffffffffffffffff")
    # Different filename, near phash → must NOT group (distinct adjacent shot).
    _add(conn, "/A/IMG_2.JPG", "0000000000000001")
    conn.commit()
    conn.close()

    from typer.testing import CliRunner
    from imagewizard.cli import app
    res = CliRunner().invoke(
        app, ["find-duplicates", "--near", "6", "--dedupe-index"])
    assert res.exit_code == 0, res.output

    conn = db.connect(cfg.db_path)
    rows = {r["path"]: r["dup_of"]
            for r in conn.execute("SELECT path, dup_of FROM files")}
    ids = {r["path"]: r["id"]
           for r in conn.execute("SELECT path, id FROM files")}
    conn.close()

    # Exactly one of the two close same-name copies is hidden as a dup of
    # the other; the keeper (shortest/first path) stays visible.
    hidden = [p for p, d in rows.items() if d is not None]
    assert hidden == ["/B/IMG_1.JPG"]
    assert rows["/B/IMG_1.JPG"] == ids["/A/IMG_1.JPG"]
    # Far-phash same-name and near-phash different-name stay visible.
    assert rows["/C/IMG_1.JPG"] is None
    assert rows["/A/IMG_2.JPG"] is None
