"""Face clustering via HDBSCAN.

Runs as a post-processing step after all faces have been detected and
embedded. Groups similar faces into clusters and preserves user-assigned
names across re-runs.

Performance notes:
- Embeddings are loaded in a single bulk query (not one per face).
- HDBSCAN runs with all cores (`core_dist_n_jobs=-1`).
- DB writes are batched with `executemany` inside a single transaction.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import time

import numpy as np
import typer
from rich.console import Console

from . import config, db

log = logging.getLogger(__name__)

EMB_DIM = 512


def _load_embeddings(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    """Load all face embeddings in one query. Returns (face_ids, matrix)."""
    # Single join — no N+1 queries
    rows = conn.execute(
        """SELECT f.id, v.embedding
           FROM faces f
           JOIN vec_faces v ON v.rowid = f.id
           ORDER BY f.id"""
    ).fetchall()

    if not rows:
        return [], np.empty((0, EMB_DIM), dtype=np.float32)

    face_ids = []
    # Pre-allocate matrix
    embeddings = np.empty((len(rows), EMB_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        face_ids.append(row["id"])
        embeddings[i] = np.frombuffer(row["embedding"], dtype=np.float32)

    return face_ids, embeddings


def cluster_faces(
    conn: sqlite3.Connection,
    min_cluster_size: int = 3,
    merge_threshold: float = 0.35,
) -> dict[str, int]:
    """Run HDBSCAN on all face embeddings and write cluster IDs."""
    import hdbscan

    console = Console(stderr=True)
    t0 = time.monotonic()

    console.print("[dim]loading face embeddings...[/dim]", end=" ")
    face_ids, embeddings = _load_embeddings(conn)
    console.print(f"[green]{len(face_ids)} faces[/green]")

    if len(face_ids) < min_cluster_size:
        return {"faces": len(face_ids), "clusters": 0}

    # L2 normalize (vectorized)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = embeddings / norms

    console.print("[dim]running HDBSCAN...[/dim]", end=" ")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(normed)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    console.print(f"[green]{len(unique_labels)} clusters, {(labels == -1).sum()} noise[/green]")

    # Load existing named clusters for stable ID assignment
    existing = {}
    for row in conn.execute(
        "SELECT cluster_id, centroid, person_name FROM face_clusters"
    ).fetchall():
        c = np.frombuffer(row["centroid"], dtype=np.float32).copy()
        n = np.linalg.norm(c)
        existing[row["cluster_id"]] = {
            "centroid": c / n if n > 0 else c,
            "name": row["person_name"],
        }

    # Pre-compute all centroids at once using vectorized ops
    label_to_cluster: dict[int, int] = {}
    next_id = max(existing.keys(), default=0) + 1

    # Build centroid matrix for existing clusters for batch distance calc
    if existing:
        exist_ids = list(existing.keys())
        exist_centroids = np.stack([existing[k]["centroid"] for k in exist_ids])
    else:
        exist_ids = []
        exist_centroids = np.empty((0, EMB_DIM), dtype=np.float32)

    cluster_upserts = []   # (cluster_id, centroid_bytes, face_count, person_name_or_None)
    cluster_inserts = []   # same format, for new clusters

    for lbl in sorted(unique_labels):
        mask = labels == lbl
        centroid = normed[mask].mean(axis=0)
        n = np.linalg.norm(centroid)
        centroid = centroid / n if n > 0 else centroid
        count = int(mask.sum())
        centroid_bytes = centroid.astype(np.float32).tobytes()

        # Find nearest existing cluster
        best_id = None
        if len(exist_centroids) > 0:
            dists = np.linalg.norm(exist_centroids - centroid, axis=1)
            min_idx = dists.argmin()
            if dists[min_idx] < merge_threshold:
                best_id = exist_ids[min_idx]

        if best_id is not None:
            label_to_cluster[lbl] = best_id
            cluster_upserts.append((centroid_bytes, count, best_id))
        else:
            label_to_cluster[lbl] = next_id
            cluster_inserts.append((next_id, centroid_bytes, count))
            next_id += 1

    # --- Batch DB writes in a single transaction ---
    console.print("[dim]writing clusters to DB...[/dim]", end=" ")

    conn.execute("BEGIN")
    try:
        # Update existing clusters
        if cluster_upserts:
            conn.executemany(
                "UPDATE face_clusters SET centroid=?, face_count=? WHERE cluster_id=?",
                cluster_upserts,
            )

        # Insert new clusters
        if cluster_inserts:
            conn.executemany(
                "INSERT INTO face_clusters (cluster_id, centroid, face_count) VALUES (?, ?, ?)",
                cluster_inserts,
            )

        # Batch update faces — build (cluster_id, face_id) pairs
        face_updates = []
        for i, fid in enumerate(face_ids):
            cid = label_to_cluster.get(labels[i])  # None for noise (-1)
            face_updates.append((cid, fid))

        conn.executemany(
            "UPDATE faces SET cluster_id=? WHERE id=?",
            face_updates,
        )

        # Propagate person_name from face_clusters → faces (batch)
        named = conn.execute(
            "SELECT cluster_id, person_name FROM face_clusters WHERE person_name IS NOT NULL"
        ).fetchall()
        if named:
            conn.executemany(
                "UPDATE faces SET person_name=? WHERE cluster_id=?",
                [(r["person_name"], r["cluster_id"]) for r in named],
            )

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    elapsed = time.monotonic() - t0
    n_clusters = len(set(label_to_cluster.values()))
    console.print(f"[green]done[/green] ({elapsed:.1f}s)")

    return {
        "faces": len(face_ids),
        "clusters": n_clusters,
        "noise": int((labels == -1).sum()),
        "seconds": round(elapsed, 1),
    }


# ---- CLI registration ----

def register(parent: typer.Typer) -> None:
    @parent.command(name="cluster-faces")
    def cmd_cluster(
        min_size: int = typer.Option(3, "--min-size", help="Min faces to form a cluster."),
    ) -> None:
        """Run face clustering (HDBSCAN) on detected face embeddings."""
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        try:
            result = cluster_faces(conn, min_cluster_size=min_size)
            console = Console()
            for k, v in result.items():
                console.print(f"  {k}: {v}")
        finally:
            conn.close()
