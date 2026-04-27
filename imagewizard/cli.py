"""image-wizard CLI entrypoint.

Subcommands are thin wrappers that delegate to modules. Keep business logic
out of this file — it exists only to parse arguments and print results.
"""

from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import config, db

# Dump a Python stack trace on native crashes (SIGSEGV, SIGABRT, etc) to
# both stderr AND a durable file. Stderr alone isn't reliable: Rich's
# progress display continuously rewrites the terminal and a multi-line
# fault dump can flash and disappear before the user sees it. The file
# version always survives.
#
# We also dump on SIGTERM/SIGUSR1 so kernel-initiated kills (memory
# pressure, supervisor termination) leave a Python-level breadcrumb of
# what every thread was doing. SIGKILL can't be intercepted — for that
# kind of death look at `image-wizard last-crash` and `log show ...`.
def _enable_faulthandler() -> None:
    try:
        from . import config as _cfg
        log_dir = _cfg.load().cache_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        f = open(log_dir / "faulthandler.log", "a", buffering=1)
    except Exception:
        f = sys.stderr
    faulthandler.enable(file=f, all_threads=True)
    # Best-effort: dump on receipt of SIGTERM (graceful kill) and SIGUSR1
    # (so the user can `kill -USR1 <pid>` to snapshot a hung process).
    import signal
    for sig in (signal.SIGTERM, getattr(signal, "SIGUSR1", None)):
        if sig is None:
            continue
        try:
            faulthandler.register(sig, file=f, all_threads=True, chain=False)
        except (ValueError, OSError):
            pass

_enable_faulthandler()

app = typer.Typer(
    name="image-wizard",
    help="Local, on-device photo indexer with EXIF, YOLO, faces, and CLIP search.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init() -> None:
    """Create the database and required directories."""
    cfg = config.load()
    db.init(cfg.db_path)
    console.print(f"[green]initialized[/green] {cfg.db_path}")
    console.print(f"  cache: {cfg.cache_dir}")


@app.command()
def stats() -> None:
    """Show counts and basic library statistics."""
    cfg = config.load()
    conn = db.connect(cfg.db_path)
    try:
        counts = {
            "files":       conn.execute("SELECT COUNT(*) FROM files WHERE missing=0").fetchone()[0],
            "with_meta":   conn.execute("SELECT COUNT(*) FROM files WHERE meta_done=1 AND missing=0").fetchone()[0],
            "with_yolo":   conn.execute("SELECT COUNT(*) FROM files WHERE yolo_done=1 AND missing=0").fetchone()[0],
            "with_faces":  conn.execute("SELECT COUNT(*) FROM files WHERE faces_done=1 AND missing=0").fetchone()[0],
            "with_clip":   conn.execute("SELECT COUNT(*) FROM files WHERE clip_done=1 AND missing=0").fetchone()[0],
            "detections":  conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0],
            "faces":       conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
        }
        t = Table(title="image-wizard library", show_header=False)
        for k, v in counts.items():
            t.add_row(k, str(v))
        console.print(t)

        row = conn.execute(
            "SELECT MIN(taken_at), MAX(taken_at) FROM photo_meta WHERE taken_at IS NOT NULL"
        ).fetchone()
        if row and row[0]:
            console.print(f"date range: {row[0]} → {row[1]}")

        cams = conn.execute(
            "SELECT camera_model, COUNT(*) c FROM photo_meta "
            "WHERE camera_model IS NOT NULL GROUP BY camera_model ORDER BY c DESC LIMIT 10"
        ).fetchall()
        if cams:
            ct = Table(title="top cameras")
            ct.add_column("model")
            ct.add_column("count", justify="right")
            for r in cams:
                ct.add_row(r["camera_model"] or "?", str(r["c"]))
            console.print(ct)
    finally:
        conn.close()


# Heavier subcommands live in their own modules and register themselves with
# the app here. Each registration block is wrapped so a missing dependency
# (e.g. torch not yet installed) still leaves `init` and `stats` usable.
def _try_register(mod_path: str, attr: str = "register") -> None:
    try:
        mod = __import__(mod_path, fromlist=[attr])
        getattr(mod, attr)(app)
    except ImportError:
        pass


for _m in (
    "imagewizard.scan",
    "imagewizard.pipeline",
    "imagewizard.cluster",
    "imagewizard.search_cli",
    "imagewizard.web.app",
):
    _try_register(_m)


if __name__ == "__main__":
    app()
