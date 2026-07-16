"""Photo-orientation model CLIs: `train-orientation` and `suggest-rotations`.

Trains the self-supervised orientation CNN on the library, then scans
photos and records a suggested clockwise rotation for the ones it's
confident are stored sideways (old cameras that wrote no EXIF orientation).
Suggestions are reviewed + applied in the web UI at ``/rotations`` — never
auto-applied. Both commands work off the cached 512px thumbnails (local,
fast, already EXIF-oriented), not the originals.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import typer
from rich.console import Console

from . import config, db
from .models import orientation as orient
from .thumbs import thumb_path

log = logging.getLogger(__name__)


def _load_thumb(cache_dir, content_hash):
    """Cached thumbnail → (INPUT, INPUT, 3) uint8, or None if absent/bad."""
    from PIL import Image
    p = thumb_path(cache_dir, content_hash)
    if not p.exists():
        return None
    try:
        im = (Image.open(p).convert("RGB")
              .resize((orient.INPUT, orient.INPUT), Image.BILINEAR))
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def _load_many(cache_dir, hashes, workers: int = 8):
    """Parallel-load thumbnails. Returns (stacked_imgs, kept_indices)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda h: _load_thumb(cache_dir, h), hashes))
    imgs, idx = [], []
    for i, r in enumerate(results):
        if r is not None:
            imgs.append(r)
            idx.append(i)
    if not imgs:
        return np.empty((0, orient.INPUT, orient.INPUT, 3), np.uint8), []
    return np.stack(imgs), idx


def register(parent: typer.Typer) -> None:
    @parent.command(name="train-orientation")
    def cmd_train_orientation(
        samples: int = typer.Option(
            3000, "--samples",
            help="How many library photos to train on (assumed upright)."),
        epochs: int = typer.Option(6, "--epochs", help="Training epochs."),
    ) -> None:
        """Train the orientation model self-supervised on your library.

        Samples correctly-oriented photos (any that you haven't manually
        rotated), rotates each to all four orientations on the fly, and
        learns to predict the rotation — so it picks up the cues that make
        a photo "upright" (sky up, faces up, horizons level, text upright).
        Saves the model to the cache; re-run to retrain. Needs a GPU/MPS
        for reasonable speed (a few minutes).
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        console = Console()
        try:
            rows = conn.execute(
                """SELECT content_hash FROM files
                   WHERE missing = 0 AND kind = 'image' AND rotation = 0
                     AND content_hash != ''
                   ORDER BY RANDOM() LIMIT ?""",
                (samples * 2,),   # oversample; some thumbnails may be absent
            ).fetchall()
            console.print(f"[dim]loading up to {samples} thumbnails…[/dim]")
            imgs, _ = _load_many(cfg.cache_dir, [r["content_hash"] for r in rows])
            imgs = imgs[:samples]
            if len(imgs) < 200:
                console.print(
                    f"[red]only {len(imgs)} thumbnails available[/red] — run "
                    "`index` (or `regen-thumbs`) so there's data to train on."
                )
                raise typer.Exit(1)
            console.print(
                f"training on {len(imgs)} images, {epochs} epochs — the model "
                "learns orientation from the upright majority…"
            )
            out = orient.train(imgs, cfg.cache_dir, epochs=epochs,
                               log_fn=lambda m: console.print(f"  {m}"))
            console.print(f"[green]saved orientation model[/green] → {out}")
            console.print("Next: `image-wizard suggest-rotations`")
        finally:
            conn.close()

    @parent.command(name="suggest-rotations")
    def cmd_suggest_rotations(
        min_conf: float = typer.Option(
            0.90, "--min-conf",
            help="Only record a suggestion when the model is at least this "
                 "confident (0-1). Higher = fewer, safer suggestions."),
        camera: str = typer.Option(
            "", "--camera",
            help="Only scan photos from camera models matching this substring."),
        limit: int = typer.Option(
            0, "--limit", help="Cap photos scanned (0 = all candidates)."),
        redo: bool = typer.Option(
            False, "--redo",
            help="Re-scan photos already checked (e.g. after retraining)."),
        batch: int = typer.Option(256, "--batch", help="Inference batch size."),
    ) -> None:
        """Scan photos and suggest a rotation for the ones stored sideways.

        Runs the orientation model over photos you haven't manually rotated
        and records a suggested clockwise correction when it's confident the
        photo is non-upright. Content-based, so it catches sideways photos
        from any orientation-blind camera — not just one model. Nothing is
        applied: review and one-click accept at ``/rotations`` in the web UI.
        """
        cfg = config.load()
        console = Console()
        if not orient.available(cfg.cache_dir):
            console.print("[red]no orientation model yet[/red] — run "
                          "`image-wizard train-orientation` first.")
            raise typer.Exit(1)
        conn = db.connect(cfg.db_path)
        try:
            where = ["f.missing = 0", "f.kind = 'image'", "f.rotation = 0",
                     "f.content_hash != ''"]
            params: list = []
            if not redo:
                where.append("f.rotation_checked = 0")
            join = ""
            if camera:
                join = "JOIN photo_meta pm ON pm.file_id = f.id"
                where.append("pm.camera_model LIKE ?")
                params.append(f"%{camera}%")
            sql = (f"SELECT f.id, f.content_hash FROM files f {join} "
                   f"WHERE {' AND '.join(where)}")
            if limit:
                sql += f" LIMIT {int(limit)}"
            rows = conn.execute(sql, params).fetchall()
            console.print(f"[bold]{len(rows)} candidate photo(s)[/bold] to check")
            if not rows:
                return

            from rich.progress import (Progress, BarColumn, MofNCompleteColumn,
                                       SpinnerColumn, TimeRemainingColumn)
            suggested = checked = no_thumb = 0
            with Progress(SpinnerColumn(), BarColumn(), MofNCompleteColumn(),
                          TimeRemainingColumn(),
                          console=Console(stderr=True)) as prog:
                task = prog.add_task("scanning", total=len(rows))
                for i in range(0, len(rows), batch):
                    chunk = rows[i:i + batch]
                    imgs, idx = _load_many(
                        cfg.cache_dir, [r["content_hash"] for r in chunk])
                    no_thumb += len(chunk) - len(idx)
                    preds = (orient.predict_batch(imgs, cfg.cache_dir)
                             if len(imgs) else [])
                    pred_by_row = {idx[j]: preds[j] for j in range(len(idx))}
                    for j, r in enumerate(chunk):
                        if j in pred_by_row:
                            corr, prob = pred_by_row[j]
                            if corr != 0 and prob >= min_conf:
                                conn.execute(
                                    "UPDATE files SET rotation_checked=1, "
                                    "rotation_suggested=?, rotation_suggested_conf=? "
                                    "WHERE id=?", (corr, prob, r["id"]))
                                suggested += 1
                            else:
                                conn.execute(
                                    "UPDATE files SET rotation_checked=1, "
                                    "rotation_suggested=NULL, "
                                    "rotation_suggested_conf=NULL WHERE id=?",
                                    (r["id"],))
                            checked += 1
                        prog.advance(task)
                    conn.commit()
            msg = (f"[green]checked {checked}[/green], "
                   f"[cyan]{suggested} suggestion(s)[/cyan] (conf ≥ {min_conf})")
            if no_thumb:
                msg += f", {no_thumb} skipped (no thumbnail)"
            console.print(msg)
            if suggested:
                console.print("Review + apply at [bold]/rotations[/bold] in the web UI.")
        finally:
            conn.close()
