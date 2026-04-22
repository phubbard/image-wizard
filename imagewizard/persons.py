"""Person identity and name-epoch helpers.

A *person* is an editable identity that can span multiple HDBSCAN
`face_clusters` rows and carry multiple names over time (married name
change, nickname, anglicised spelling, etc.).

Each person owns one or more `person_names` rows, an "epoch":
``(name, start_date, end_date)``. Either bound may be ``NULL`` meaning
open-ended. The *displayed* name for a face is the epoch whose date
range covers the photo's ``taken_at``, falling back to the person's
``primary_name`` when no epoch matches (or when the photo has no date).

For render speed we keep the resolved name cached on ``faces.person_name``
and recompute it whenever an epoch is added, edited, or deleted, or
whenever a cluster's ``person_id`` changes.
"""

from __future__ import annotations

import sqlite3


# ---------- look-ups --------------------------------------------------------


def find_person_by_name(conn: sqlite3.Connection, name: str) -> int | None:
    """Resolve a name (any epoch, case-insensitive) to a person_id.

    Returns the lowest matching id so behaviour is stable when the same
    string is registered against multiple persons (shouldn't happen, but
    a name collision could result from a partial migration).
    """
    row = conn.execute(
        """SELECT person_id FROM person_names
           WHERE name = ? COLLATE NOCASE
           ORDER BY person_id
           LIMIT 1""",
        (name,),
    ).fetchone()
    return row["person_id"] if row else None


def get_person(conn: sqlite3.Connection, person_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM persons WHERE id = ?", (person_id,)
    ).fetchone()


def list_name_epochs(conn: sqlite3.Connection, person_id: int) -> list[sqlite3.Row]:
    """Return the person's name epochs in chronological order.

    Epochs with an explicit start date come first (ascending); open-start
    epochs (``start_date IS NULL``) come last so the timeline reads
    naturally top-to-bottom.
    """
    return conn.execute(
        """SELECT id, name, start_date, end_date, is_nickname
           FROM person_names
           WHERE person_id = ?
           ORDER BY (CASE WHEN start_date IS NULL THEN 1 ELSE 0 END),
                    start_date ASC""",
        (person_id,),
    ).fetchall()


def all_known_names(conn: sqlite3.Connection) -> list[str]:
    """Every distinct name across every person, for autocomplete."""
    return [
        r[0]
        for r in conn.execute(
            """SELECT DISTINCT name FROM person_names
               ORDER BY name COLLATE NOCASE"""
        ).fetchall()
    ]


# ---------- writes ----------------------------------------------------------


def create_person(conn: sqlite3.Connection, name: str) -> int:
    """Create a new person with one open-ended name epoch."""
    cur = conn.execute(
        "INSERT INTO persons (primary_name) VALUES (?)", (name,)
    )
    pid = cur.lastrowid
    conn.execute(
        "INSERT INTO person_names (person_id, name) VALUES (?, ?)",
        (pid, name),
    )
    return pid


def get_or_create_person(conn: sqlite3.Connection, name: str) -> int:
    """Resolve ``name`` to a person_id, creating one if no match exists."""
    pid = find_person_by_name(conn, name)
    if pid is not None:
        return pid
    return create_person(conn, name)


def add_name_epoch(
    conn: sqlite3.Connection,
    person_id: int,
    name: str,
    start_date: str | None,
    end_date: str | None,
    is_nickname: bool = False,
) -> int:
    """Add a name epoch. Returns the new person_names.id.

    If ``name`` already belongs to another person, that other person is
    merged into ``person_id`` (clusters reassigned, person_names rows
    moved over, persons row deleted) so we end up with one identity
    carrying both names.

    Avoids creating a duplicate epoch row when the merge already
    transferred an identical (name, start_date, end_date) tuple — the
    user often wants to *say* "this name belongs to this person" without
    realising that fact is already on file. We update the existing row
    in-place when there's a conflict on the unbounded form.
    """
    other = find_person_by_name(conn, name)
    if other is not None and other != person_id:
        merge_persons(conn, keep=person_id, drop=other)

    # If an identical-name epoch already exists on this person from the
    # merge (or a prior add), update its date bounds to the new ones
    # rather than inserting a duplicate row.
    existing = conn.execute(
        """SELECT id FROM person_names
           WHERE person_id = ? AND name = ? COLLATE NOCASE
           ORDER BY id LIMIT 1""",
        (person_id, name),
    ).fetchone()
    if existing:
        conn.execute(
            """UPDATE person_names
               SET start_date = ?, end_date = ?, is_nickname = ?
               WHERE id = ?""",
            (start_date, end_date, 1 if is_nickname else 0, existing["id"]),
        )
        return existing["id"]

    cur = conn.execute(
        """INSERT INTO person_names
           (person_id, name, start_date, end_date, is_nickname)
           VALUES (?, ?, ?, ?, ?)""",
        (person_id, name, start_date, end_date, 1 if is_nickname else 0),
    )
    return cur.lastrowid


def delete_name_epoch(
    conn: sqlite3.Connection, person_id: int, epoch_id: int
) -> None:
    """Remove a name epoch. Refuses to delete the last epoch for a person
    (callers should delete the person itself in that case)."""
    n = conn.execute(
        "SELECT COUNT(*) FROM person_names WHERE person_id = ?",
        (person_id,),
    ).fetchone()[0]
    if n <= 1:
        raise ValueError("cannot delete the only name epoch")
    conn.execute(
        "DELETE FROM person_names WHERE id = ? AND person_id = ?",
        (epoch_id, person_id),
    )


def set_primary_name(conn: sqlite3.Connection, person_id: int, name: str) -> None:
    conn.execute(
        "UPDATE persons SET primary_name = ? WHERE id = ?", (name, person_id)
    )


def merge_persons(conn: sqlite3.Connection, keep: int, drop: int) -> None:
    """Move every cluster + name epoch from ``drop`` into ``keep`` and
    delete the ``drop`` person row.

    Caller is responsible for refresh_face_names_for_person(keep) afterward.
    """
    if keep == drop:
        return
    conn.execute(
        "UPDATE face_clusters SET person_id = ? WHERE person_id = ?",
        (keep, drop),
    )
    conn.execute(
        "UPDATE person_names SET person_id = ? WHERE person_id = ?",
        (keep, drop),
    )
    conn.execute("DELETE FROM persons WHERE id = ?", (drop,))


# ---------- date-aware name lookup + cache refresh --------------------------


def _resolve_name(
    epochs: list[sqlite3.Row], primary: str, photo_date: str | None
) -> str:
    """Pick the right name from a pre-loaded list of epochs.

    ``photo_date`` is the photo's ``taken_at`` (may be ``"YYYY-MM-DD..."``
    or ``None``). Falls back to ``primary`` if no epoch matches.

    When several epochs cover the date, the most *specific* one wins:
    epochs with both bounds beat one-bound epochs beat fully-open
    epochs. This lets the user keep an open-ended "primary" epoch
    (NULL/NULL) as a fallback while adding narrower epochs that take
    precedence in their own date windows.
    """
    if not photo_date:
        return primary
    d = photo_date[:10]  # "YYYY-MM-DD"

    best_key: tuple[int, str, int] = (-1, "", -1)
    best: sqlite3.Row | None = None
    for ep in epochs:
        sd, ed = ep["start_date"], ep["end_date"]
        if sd is not None and sd > d:
            continue
        if ed is not None and ed < d:
            continue
        specificity = (1 if sd else 0) + (1 if ed else 0)
        # Tie-breakers: more recent start (so a 2012-2020 epoch beats a
        # 2000-2025 one for a 2015 photo), then most recent id.
        key = (specificity, sd or "", ep["id"])
        if key > best_key:
            best_key = key
            best = ep
    return best["name"] if best else primary


def refresh_face_names_for_person(
    conn: sqlite3.Connection, person_id: int
) -> int:
    """Recompute ``faces.person_name`` for every face linked (via cluster)
    to ``person_id``, using each face's own ``taken_at``.

    Returns the number of face rows updated. Cheap because we load the
    epoch list once and scan it per face in Python — usually <10 epochs.
    """
    p = get_person(conn, person_id)
    if not p:
        return 0
    primary = p["primary_name"]
    epochs = list_name_epochs(conn, person_id)
    rows = conn.execute(
        """SELECT fa.id AS face_id, pm.taken_at AS taken_at
           FROM faces fa
           JOIN face_clusters fc ON fc.cluster_id = fa.cluster_id
           LEFT JOIN photo_meta pm ON pm.file_id = fa.file_id
           WHERE fc.person_id = ?""",
        (person_id,),
    ).fetchall()
    updates = [
        (_resolve_name(epochs, primary, r["taken_at"]), r["face_id"])
        for r in rows
    ]
    if updates:
        conn.executemany(
            "UPDATE faces SET person_name = ? WHERE id = ?", updates
        )
    return len(updates)
