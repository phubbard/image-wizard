"""collate: build a deduplicated, date-organized tree + re-point the index."""
from __future__ import annotations


def test_collate_builds_dedup_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGEWIZARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IMAGEWIZARD_CACHE_DIR", str(tmp_path / "cache"))
    from imagewizard import config, db, scan
    cfg = config.load()
    db.init(cfg.db_path)
    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "library"
    conn = db.connect(cfg.db_path)

    def add(rel, content, taken=None, dup_of=None, missing=0):
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        h = scan.content_hash(p)
        conn.execute(
            """INSERT INTO files(path, content_hash, size, mtime, indexed_at,
                                 missing, kind, dup_of)
               VALUES(?,?,?,?,?,?,'image',?)""",
            (str(p), h, p.stat().st_size, 0.0, 0.0, missing, dup_of),
        )
        fid = conn.execute(
            "SELECT id FROM files WHERE path=?", (str(p),)).fetchone()[0]
        conn.execute(
            "INSERT INTO photo_meta(file_id, taken_at) VALUES(?,?)", (fid, taken))
        return fid, h

    a_id, _ = add("a.jpg", b"AAAA", taken="2003-04-24 10:00:00")
    add("u.jpg", b"UUUU", taken=None)                         # undated
    _, c1h = add("c1/IMG_1.JPG", b"ONE", taken="2005-01-01 00:00:00")
    _, c2h = add("c2/IMG_1.JPG", b"TWO", taken="2005-01-01 00:00:00")  # collision
    dup_id, _ = add("dup.jpg", b"DUP", taken="2003-04-24 10:00:00", dup_of=a_id)
    add("gone.jpg", b"GONE", taken="2003-04-24 10:00:00", missing=1)
    conn.commit()
    conn.close()

    from typer.testing import CliRunner
    from imagewizard.cli import app
    res = CliRunner().invoke(
        app, ["collate", str(dst), "--apply", "--workers", "2"])
    assert res.exit_code == 0, res.output

    # dated → YYYY/MM/DD; undated → undated/; content preserved
    assert (dst / "2003/04/24/a.jpg").read_bytes() == b"AAAA"
    assert (dst / "undated/u.jpg").read_bytes() == b"UUUU"
    # same date+name, different photos → both hash-suffixed, both copied
    assert (dst / f"2005/01/01/IMG_1-{c1h[:8]}.JPG").read_bytes() == b"ONE"
    assert (dst / f"2005/01/01/IMG_1-{c2h[:8]}.JPG").read_bytes() == b"TWO"
    # duplicates + missing are NOT collated
    assert not (dst / "2003/04/24/dup.jpg").exists()
    assert not (dst / "2003/04/24/gone.jpg").exists()

    # index re-pointed for collated files; the (excluded) dup left alone
    conn = db.connect(cfg.db_path)
    paths = {r["id"]: r["path"] for r in conn.execute("SELECT id, path FROM files")}
    conn.close()
    assert paths[a_id] == str(dst / "2003/04/24/a.jpg")
    assert paths[dup_id] == str(src / "dup.jpg")

    # resumable: a second run copies nothing new
    res2 = CliRunner().invoke(
        app, ["collate", str(dst), "--apply", "--workers", "2"])
    assert res2.exit_code == 0
    assert "copied 0" in res2.output and "skipped 4" in res2.output
