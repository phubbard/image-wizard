"""Face clustering via HDBSCAN.

Runs as a post-processing step after faces have been detected and embedded.
Groups similar faces into clusters and preserves user-assigned names across
re-runs.

Two modes:

* **Incremental** (default): only faces with `cluster_id IS NULL` are
  considered. Each new face is first matched (vectorized) against the
  centroids of existing clusters — if the nearest is within `merge_threshold`
  the face inherits that cluster. Only the remaining unmatched faces are fed
  to HDBSCAN. This keeps re-runs cheap as the corpus grows: the expensive
  O(N^2)-ish step only sees *new* data.

* **Full** (`--full`): load every face, rebuild clusters from scratch via
  HDBSCAN, and map each new cluster onto the closest existing cluster so
  user-assigned names survive. Use this after big data changes or if the
  clustering has drifted badly.

Both paths also run a name-inheritance pass: if a cluster has no
`person_name` but ≥2 of its member faces already carry the same user-assigned
name (from a prior session / merge), the cluster adopts that name. This
handles the "kid ages and HDBSCAN splits the cluster" case.
"""

from __future__ import annotations

import logging
import sqlite3
import time

import numpy as np
import typer
from rich.console import Console

from . import config, db

log = logging.getLogger(__name__)

EMB_DIM = 512


# ---------- helpers ---------------------------------------------------------


def _l2norm(m: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize. Safe against zero / NaN / Inf rows.

    Zero-magnitude rows are kept as all-zero (not divided by a tiny
    epsilon, which would blow them up to huge values that overflow in
    later float32 matmuls). NaN/Inf values are replaced with zero
    before normalising so a single corrupt embedding can't poison the
    whole matrix.
    """
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    safe = np.where(n > 0, n, 1.0)
    out = m / safe
    # Rows that were originally zero-magnitude → stay zero.
    out = np.where(n > 0, out, 0.0)
    return out


def _load_embeddings_where(
    conn: sqlite3.Connection, where_clause: str = ""
) -> tuple[list[int], np.ndarray]:
    """Load face embeddings matching an optional WHERE clause.

    The caller owns the WHERE text; we just paste it in. No parameters are
    accepted — this is a fixed-input helper, not user-facing.
    """
    rows = conn.execute(
        f"""SELECT f.id, v.embedding
            FROM faces f
            JOIN vec_faces v ON v.rowid = f.id
            {where_clause}
            ORDER BY f.id"""
    ).fetchall()

    if not rows:
        return [], np.empty((0, EMB_DIM), dtype=np.float32)

    face_ids: list[int] = []
    embeddings = np.empty((len(rows), EMB_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        face_ids.append(row["id"])
        embeddings[i] = np.frombuffer(row["embedding"], dtype=np.float32)
    return face_ids, embeddings


def _load_existing_clusters(
    conn: sqlite3.Connection,
) -> tuple[list[int], np.ndarray, dict[int, str]]:
    """Return (cluster_ids, L2-normed centroid matrix, {cid: person_name}).

    Degenerate centroids (all-zero, NaN, Inf) are skipped entirely —
    matching a new face against a zero vector is meaningless and the
    bogus values cause overflow warnings in the downstream matmul.
    """
    ids: list[int] = []
    cents: list[np.ndarray] = []
    names: dict[int, str] = {}
    for row in conn.execute(
        "SELECT cluster_id, centroid, person_name FROM face_clusters"
    ).fetchall():
        c = np.frombuffer(row["centroid"], dtype=np.float32).copy()
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        n = np.linalg.norm(c)
        if not np.isfinite(n) or n <= 0:
            # Unusable — skip this cluster during the matching pass.
            continue
        c = c / n
        ids.append(row["cluster_id"])
        cents.append(c)
        if row["person_name"]:
            names[row["cluster_id"]] = row["person_name"]
    if cents:
        centroids = np.stack(cents)
    else:
        centroids = np.empty((0, EMB_DIM), dtype=np.float32)
    return ids, centroids, names


def _recompute_cluster_centroids(
    conn: sqlite3.Connection, cluster_ids: set[int]
) -> None:
    """Upsert centroid + face_count rows for the given clusters."""
    for cid in cluster_ids:
        rows = conn.execute(
            """SELECT v.embedding
               FROM faces f
               JOIN vec_faces v ON v.rowid = f.id
               WHERE f.cluster_id = ?""",
            (cid,),
        ).fetchall()
        if not rows:
            # Cluster has no members anymore — drop it.
            conn.execute("DELETE FROM face_clusters WHERE cluster_id=?", (cid,))
            continue
        mat = np.empty((len(rows), EMB_DIM), dtype=np.float32)
        for i, r in enumerate(rows):
            mat[i] = np.frombuffer(r["embedding"], dtype=np.float32)
        normed = _l2norm(mat)
        centroid = normed.mean(axis=0)
        n = np.linalg.norm(centroid)
        if n > 0:
            centroid = centroid / n
        conn.execute(
            """INSERT INTO face_clusters (cluster_id, centroid, face_count)
               VALUES (?, ?, ?)
               ON CONFLICT(cluster_id) DO UPDATE
                   SET centroid=excluded.centroid,
                       face_count=excluded.face_count""",
            (cid, centroid.astype(np.float32).tobytes(), len(rows)),
        )


def _purge_empty_clusters(conn: sqlite3.Connection) -> int:
    """Delete `face_clusters` rows that no longer have any member faces.

    Clusters get orphaned when every source photo they reference is later
    deleted (CASCADE removes the `faces` rows but the `face_clusters` row
    persists, leaving a phantom card on the Faces page with a stale
    face_count and no representative thumbnail).
    """
    cur = conn.execute(
        """DELETE FROM face_clusters
           WHERE cluster_id NOT IN (
               SELECT DISTINCT cluster_id FROM faces
               WHERE cluster_id IS NOT NULL
           )"""
    )
    return cur.rowcount or 0


def _inherit_names(conn: sqlite3.Connection) -> None:
    """For unnamed clusters, adopt a name that ≥2 member faces already carry."""
    unnamed = conn.execute(
        "SELECT cluster_id FROM face_clusters WHERE person_name IS NULL"
    ).fetchall()
    for uc in unnamed:
        cid = uc["cluster_id"]
        row = conn.execute(
            """SELECT person_name, COUNT(*) AS cnt
               FROM faces
               WHERE cluster_id = ? AND person_name IS NOT NULL
               GROUP BY person_name
               ORDER BY cnt DESC
               LIMIT 1""",
            (cid,),
        ).fetchone()
        if row and row["cnt"] >= 2:
            conn.execute(
                "UPDATE face_clusters SET person_name=? WHERE cluster_id=?",
                (row["person_name"], cid),
            )

    # Propagate cluster names → faces (only where the face has no name yet).
    named = conn.execute(
        "SELECT cluster_id, person_name FROM face_clusters WHERE person_name IS NOT NULL"
    ).fetchall()
    if named:
        conn.executemany(
            "UPDATE faces SET person_name=? WHERE cluster_id=? AND person_name IS NULL",
            [(r["person_name"], r["cluster_id"]) for r in named],
        )


# ---------- incremental path (default) --------------------------------------


def cluster_faces_incremental(
    conn: sqlite3.Connection,
    min_cluster_size: int = 3,
    merge_threshold: float = 0.35,
) -> dict[str, int | float]:
    """Cluster only faces whose `cluster_id IS NULL`.

    Workflow:
      1. Load new faces.
      2. Vectorized nearest-centroid match against existing clusters.
      3. Faces within `merge_threshold` → inherit that cluster.
      4. Remaining faces → HDBSCAN.
      5. For each HDBSCAN cluster, try once more to merge with existing.
      6. Recompute centroids + counts for every touched cluster.
    """
    import hdbscan

    console = Console(stderr=True)
    t0 = time.monotonic()

    console.print("[dim]loading unclustered faces...[/dim]", end=" ")
    face_ids, embeddings = _load_embeddings_where(
        conn, "WHERE f.cluster_id IS NULL"
    )
    console.print(f"[green]{len(face_ids)} new faces[/green]")

    if not face_ids:
        return {
            "faces": 0,
            "matched_to_existing": 0,
            "new_clusters": 0,
            "noise": 0,
            "seconds": round(time.monotonic() - t0, 1),
        }

    normed = _l2norm(embeddings)

    existing_ids, existing_centroids, _existing_names = _load_existing_clusters(conn)

    # --- Step 1: per-face nearest-centroid match (vectorized) --------------
    matched_face_to_cluster: dict[int, int] = {}
    unmatched_mask = np.ones(len(face_ids), dtype=bool)

    if len(existing_centroids) > 0:
        console.print(
            f"[dim]matching against {len(existing_centroids)} existing clusters...[/dim]",
            end=" ",
        )
        # All rows are L2-normed → ||a-b||^2 = 2 - 2·(a·b)
        dots = normed @ existing_centroids.T  # (N_new, N_existing)
        d2 = np.maximum(0.0, 2.0 - 2.0 * dots)
        dists = np.sqrt(d2)
        best_idx = dists.argmin(axis=1)
        best_dist = dists[np.arange(len(face_ids)), best_idx]
        matched = best_dist < merge_threshold

        for i in np.where(matched)[0]:
            matched_face_to_cluster[face_ids[i]] = existing_ids[int(best_idx[i])]
        unmatched_mask = ~matched
        console.print(f"[green]{int(matched.sum())} matched[/green]")

    # --- Step 2: HDBSCAN on leftovers -------------------------------------
    leftover_idx = np.where(unmatched_mask)[0]
    n_leftover = len(leftover_idx)

    new_cluster_assignments: dict[int, int] = {}
    noise_count = 0
    new_clusters_created = 0

    if n_leftover >= min_cluster_size:
        console.print(
            f"[dim]running HDBSCAN on {n_leftover} leftover faces...[/dim]",
            end=" ",
        )
        leftover_normed = normed[leftover_idx]
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(leftover_normed)

        unique_labels = set(int(x) for x in labels)
        unique_labels.discard(-1)
        noise_count = int((labels == -1).sum())
        console.print(
            f"[green]{len(unique_labels)} clusters, {noise_count} noise[/green]"
        )

        next_id = max(existing_ids, default=0) + 1
        for lbl in sorted(unique_labels):
            mask = labels == lbl
            centroid = leftover_normed[mask].mean(axis=0)
            n = np.linalg.norm(centroid)
            if n > 0:
                centroid = centroid / n

            # Does this new cluster itself sit near an existing one?
            best_cid: int | None = None
            if len(existing_centroids) > 0:
                dd = np.linalg.norm(existing_centroids - centroid, axis=1)
                j = int(dd.argmin())
                if dd[j] < merge_threshold:
                    best_cid = existing_ids[j]

            if best_cid is None:
                best_cid = next_id
                next_id += 1
                new_clusters_created += 1

            for local_i in np.where(mask)[0]:
                face_pos = int(leftover_idx[int(local_i)])
                new_cluster_assignments[face_ids[face_pos]] = best_cid
    else:
        if n_leftover > 0:
            console.print(
                f"[dim]skipping HDBSCAN: {n_leftover} leftover < min_cluster_size={min_cluster_size}[/dim]"
            )

    # --- Step 3: write back ------------------------------------------------
    console.print("[dim]writing to DB...[/dim]", end=" ")
    conn.execute("BEGIN")
    try:
        if matched_face_to_cluster:
            conn.executemany(
                "UPDATE faces SET cluster_id=? WHERE id=?",
                [(cid, fid) for fid, cid in matched_face_to_cluster.items()],
            )
        if new_cluster_assignments:
            conn.executemany(
                "UPDATE faces SET cluster_id=? WHERE id=?",
                [(cid, fid) for fid, cid in new_cluster_assignments.items()],
            )

        touched = set(matched_face_to_cluster.values()) | set(
            new_cluster_assignments.values()
        )
        _recompute_cluster_centroids(conn, touched)
        _purge_empty_clusters(conn)
        _inherit_names(conn)

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    elapsed = time.monotonic() - t0
    console.print(f"[green]done[/green] ({elapsed:.1f}s)")

    return {
        "faces": len(face_ids),
        "matched_to_existing": len(matched_face_to_cluster),
        "new_clusters": new_clusters_created,
        "noise": noise_count,
        "seconds": round(elapsed, 1),
    }


# ---------- full rebuild path -----------------------------------------------


def cluster_faces_full(
    conn: sqlite3.Connection,
    min_cluster_size: int = 3,
    merge_threshold: float = 0.35,
) -> dict[str, int | float]:
    """Run HDBSCAN on every face. Use after big data changes."""
    import hdbscan

    console = Console(stderr=True)
    t0 = time.monotonic()

    console.print("[dim]loading all face embeddings...[/dim]", end=" ")
    face_ids, embeddings = _load_embeddings_where(conn)
    console.print(f"[green]{len(face_ids)} faces[/green]")

    if len(face_ids) < min_cluster_size:
        return {"faces": len(face_ids), "clusters": 0, "noise": 0, "seconds": 0.0}

    normed = _l2norm(embeddings)

    console.print("[dim]running HDBSCAN...[/dim]", end=" ")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(normed)

    unique_labels = set(int(x) for x in labels)
    unique_labels.discard(-1)
    noise_count = int((labels == -1).sum())
    console.print(
        f"[green]{len(unique_labels)} clusters, {noise_count} noise[/green]"
    )

    existing_ids, existing_centroids, _names = _load_existing_clusters(conn)
    next_id = max(existing_ids, default=0) + 1
    label_to_cluster: dict[int, int] = {}

    for lbl in sorted(unique_labels):
        mask = labels == lbl
        centroid = normed[mask].mean(axis=0)
        n = np.linalg.norm(centroid)
        if n > 0:
            centroid = centroid / n

        best_cid: int | None = None
        if len(existing_centroids) > 0:
            dd = np.linalg.norm(existing_centroids - centroid, axis=1)
            j = int(dd.argmin())
            if dd[j] < merge_threshold:
                best_cid = existing_ids[j]
        if best_cid is None:
            best_cid = next_id
            next_id += 1
        label_to_cluster[lbl] = best_cid

    console.print("[dim]writing to DB...[/dim]", end=" ")
    conn.execute("BEGIN")
    try:
        # Full rebuild: clear face assignments first, then reassign.
        # (We leave face_clusters rows alone so names survive; centroids get
        # recomputed below.)
        conn.execute("UPDATE faces SET cluster_id=NULL")

        face_updates: list[tuple[int | None, int]] = []
        for i, fid in enumerate(face_ids):
            cid = label_to_cluster.get(int(labels[i]))  # None for noise
            face_updates.append((cid, fid))
        conn.executemany(
            "UPDATE faces SET cluster_id=? WHERE id=?",
            face_updates,
        )

        touched = set(int(v) for v in label_to_cluster.values())
        # Also touch any pre-existing clusters that are now empty so their
        # counts update (or rows get dropped).
        for cid in existing_ids:
            touched.add(cid)
        _recompute_cluster_centroids(conn, touched)
        _purge_empty_clusters(conn)
        _inherit_names(conn)

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    elapsed = time.monotonic() - t0
    console.print(f"[green]done[/green] ({elapsed:.1f}s)")

    return {
        "faces": len(face_ids),
        "clusters": len(set(label_to_cluster.values())),
        "noise": noise_count,
        "seconds": round(elapsed, 1),
    }


# ---------- public entry point ----------------------------------------------


def cluster_faces(
    conn: sqlite3.Connection,
    min_cluster_size: int = 3,
    merge_threshold: float = 0.35,
    full: bool = False,
) -> dict[str, int | float]:
    """Dispatch to the incremental or full clustering path."""
    if full:
        return cluster_faces_full(
            conn,
            min_cluster_size=min_cluster_size,
            merge_threshold=merge_threshold,
        )
    return cluster_faces_incremental(
        conn,
        min_cluster_size=min_cluster_size,
        merge_threshold=merge_threshold,
    )


# ---------- CLI registration ------------------------------------------------


def register(parent: typer.Typer) -> None:
    @parent.command(name="cluster-faces")
    def cmd_cluster(
        min_size: int = typer.Option(
            3, "--min-size", help="Min faces to form a cluster."
        ),
        full: bool = typer.Option(
            False,
            "--full",
            help="Rebuild all clusters from scratch (slow). Default is "
            "incremental: only new (unclustered) faces are processed.",
        ),
    ) -> None:
        """Run face clustering (HDBSCAN) on detected face embeddings.

        By default this is incremental — only faces that don't yet belong to a
        cluster are processed, so re-runs after a small `index` pass are fast.
        Pass `--full` to rebuild every cluster from scratch.
        """
        cfg = config.load()
        conn = db.connect(cfg.db_path)
        try:
            result = cluster_faces(conn, min_cluster_size=min_size, full=full)
            db.set_meta(conn, "last_cluster_at", str(time.time()))
            console = Console()
            mode = "full" if full else "incremental"
            console.print(f"[bold]mode:[/bold] {mode}")
            for k, v in result.items():
                console.print(f"  {k}: {v}")
        finally:
            conn.close()
