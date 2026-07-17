"""export: filtered copy of a subset (e.g. for Apple Photos import)."""
from __future__ import annotations


def test_export_filters_and_leaves_index_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGEWIZARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGEWIZARD_CACHE_DIR", str(tmp_path / "cache"))
    from imagewizard import config, db, scan
    cfg = config.load()
    db.init(cfg.db_path)
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "out"
    conn = db.connect(cfg.db_path)

    def add(name, content, taken, camera, dup_of=None):
        p = src / name
        p.write_bytes(content)
        h = scan.content_hash(p)
        conn.execute(
            """INSERT INTO files(path, content_hash, size, mtime, indexed_at,
                                 missing, kind, dup_of)
               VALUES(?,?,?,?,?,0,'image',?)""",
            (str(p), h, p.stat().st_size, 0.0, 0.0, dup_of),
        )
        fid = conn.execute(
            "SELECT id FROM files WHERE path=?", (str(p),)).fetchone()[0]
        conn.execute(
            "INSERT INTO photo_meta(file_id, taken_at, camera_model) "
            "VALUES(?,?,?)", (fid, taken, camera))
        return fid

    keep = add("k.jpg", b"KEEP", "2010-05-01 00:00:00", "Apple iPhone 4")
    add("wrongcam.jpg", b"X", "2010-05-01 00:00:00", "Canon PowerShot")   # camera
    add("toolate.jpg", b"Y", "2015-01-01 00:00:00", "Apple iPhone 6")     # date
    add("dup.jpg", b"D", "2010-05-01 00:00:00", "Apple iPhone 4", dup_of=keep)
    conn.commit()
    conn.close()

    from typer.testing import CliRunner
    from imagewizard.cli import app
    res = CliRunner().invoke(
        app, ["export", str(out), "--camera", "iPhone", "--before", "2012",
              "--apply", "--workers", "2"])
    assert res.exit_code == 0, res.output

    # Only the matching, visible photo is exported.
    assert (out / "2010/05/01/k.jpg").read_bytes() == b"KEEP"
    assert not (out / "2010/05/01/wrongcam.jpg").exists()
    assert not (out / "2015/01/01/toolate.jpg").exists()
    assert not (out / "2010/05/01/dup.jpg").exists()

    # export must NOT re-point the index (unlike collate).
    conn = db.connect(cfg.db_path)
    kpath = conn.execute(
        "SELECT path FROM files WHERE id=?", (keep,)).fetchone()[0]
    conn.close()
    assert kpath == str(src / "k.jpg")


def test_export_flat(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGEWIZARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGEWIZARD_CACHE_DIR", str(tmp_path / "cache"))
    from imagewizard import config, db, scan
    cfg = config.load()
    db.init(cfg.db_path)
    src = tmp_path / "src"
    src.mkdir()
    out = tmp_path / "flat"
    conn = db.connect(cfg.db_path)
    p = src / "photo.jpg"
    p.write_bytes(b"Z")
    conn.execute(
        """INSERT INTO files(path, content_hash, size, mtime, indexed_at,
                             missing, kind) VALUES(?,?,?,?,?,0,'image')""",
        (str(p), scan.content_hash(p), 1, 0.0, 0.0))
    fid = conn.execute("SELECT id FROM files").fetchone()[0]
    conn.execute("INSERT INTO photo_meta(file_id, taken_at) VALUES(?,?)",
                 (fid, "2010-05-01 00:00:00"))
    conn.commit()
    conn.close()

    from typer.testing import CliRunner
    from imagewizard.cli import app
    res = CliRunner().invoke(
        app, ["export", str(out), "--flat", "--apply"])
    assert res.exit_code == 0, res.output
    assert (out / "photo.jpg").read_bytes() == b"Z"   # flat, no date subdir
