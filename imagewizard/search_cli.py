"""CLIP text-to-image search CLI command."""

from __future__ import annotations

import struct

import typer
from rich.console import Console

from . import config, db


def search_clip(
    conn: "db.sqlite3.Connection",
    query: str,
    k: int = 20,
) -> list[dict]:
    """Embed a text query and return the k nearest images."""
    from .models.clip import embed_text

    vec = embed_text(query)
    vec_bytes = struct.pack(f"{len(vec)}f", *vec.tolist())

    rows = conn.execute(
        """SELECT v.rowid AS file_id, v.distance, f.path
           FROM vec_clip v
           JOIN files f ON f.id = v.rowid
           WHERE v.embedding MATCH ? AND k = ?
           ORDER BY v.distance""",
        (vec_bytes, k),
    ).fetchall()

    return [{"file_id": r["file_id"], "distance": r["distance"], "path": r["path"]} for r in rows]


# ---- CLI registration ----

def register(parent: typer.Typer) -> None:
    @parent.command(name="search")
    def cmd_search(
        query: str = typer.Argument(..., help="Free-text query (e.g. 'dog on a beach')."),
        k: int = typer.Option(20, "--k", "-k", help="Number of results."),
    ) -> None:
        """Search photos by text using CLIP embeddings."""
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        try:
            results = search_clip(conn, query, k)
            console = Console()
            if not results:
                console.print("[yellow]no results[/yellow]")
                return
            for r in results:
                console.print(f"  {r['distance']:.4f}  {r['path']}")
        finally:
            conn.close()
