"""FastAPI web application for browsing the photo index."""

from __future__ import annotations

import datetime as _dt
import logging
import math
import struct
from pathlib import Path

log = logging.getLogger(__name__)


def _fmt_bytes(n: int) -> str:
    """1.2 GB / 456 MB / ... for display."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < step:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= step
    return f"{n:.1f} PB"


def _fmt_ts(ts: str | None) -> str | None:
    """Epoch-string → 'YYYY-MM-DD HH:MM' for display."""
    if not ts:
        return None
    try:
        return _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None

import typer
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import config, db
from ..thumbs import thumb_path

WEB_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=WEB_DIR / "templates")


def create_app(cfg: config.Config | None = None) -> FastAPI:
    if cfg is None:
        cfg = config.load()

    app = FastAPI(title="image-wizard")
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    def get_conn():
        return db.connect(cfg.db_path)

    # ---- Routes ----

    def _parse_months(raw: str) -> list[str]:
        """'1,3,12' → ['01', '03', '12']. Silently drops junk."""
        out: list[str] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                n = int(tok)
            except ValueError:
                continue
            if 1 <= n <= 12:
                out.append(f"{n:02d}")
        # dedupe, preserve order
        seen: set[str] = set()
        result: list[str] = []
        for m in out:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result

    def _build_timeline_where(year: str, months: list[str]) -> tuple[str, list]:
        where = "f.missing = 0"
        params: list = []
        if year:
            where += " AND pm.taken_at LIKE ?"
            params.append(f"{year}%")
        if months:
            placeholders = ",".join("?" * len(months))
            where += f" AND SUBSTR(pm.taken_at, 6, 2) IN ({placeholders})"
            params.extend(months)
        return where, params

    @app.get("/", response_class=HTMLResponse)
    async def timeline(
        request: Request,
        page: int = Query(0, ge=0),
        year: str = Query("", description="Filter by year"),
        months: str = Query("", description="Comma-separated months (1-12)"),
    ):
        conn = get_conn()
        try:
            per_page = 60
            sel_months = _parse_months(months)

            # Available years for nav bar
            years = [r[0] for r in conn.execute(
                """SELECT DISTINCT SUBSTR(taken_at, 1, 4) AS yr
                   FROM photo_meta WHERE taken_at IS NOT NULL
                   ORDER BY yr DESC"""
            ).fetchall()]

            # Months that have photos in the selected year (for the month bar)
            available_months: list[str] = []
            if year:
                available_months = [r[0] for r in conn.execute(
                    """SELECT DISTINCT SUBSTR(taken_at, 6, 2) AS m
                       FROM photo_meta
                       WHERE taken_at LIKE ?
                       ORDER BY m""",
                    (f"{year}%",),
                ).fetchall()]

            where, params = _build_timeline_where(year, sel_months)

            total = conn.execute(
                f"""SELECT COUNT(*) FROM files f
                    LEFT JOIN photo_meta pm ON pm.file_id = f.id
                    WHERE {where}""",
                params,
            ).fetchone()[0]

            load_count = (page + 1) * per_page
            rows = conn.execute(
                f"""SELECT f.id, f.path, f.content_hash, f.width, f.height,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM files f
                   LEFT JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE {where}
                   ORDER BY COALESCE(pm.taken_at, f.mtime) DESC
                   LIMIT ?""",
                params + [load_count],
            ).fetchall()
            has_next = load_count < total
            return TEMPLATES.TemplateResponse(request, "timeline.html", {
                "photos": rows,
                "page": page,
                "has_next": has_next,
                "total": total,
                "years": years,
                "year": year,
                "available_months": available_months,
                "selected_months": sel_months,
                "months_qs": ",".join(sel_months),
            })
        finally:
            conn.close()

    @app.get("/timeline-page", response_class=HTMLResponse)
    async def timeline_page(
        request: Request,
        page: int = Query(0, ge=0),
        year: str = Query(""),
        months: str = Query(""),
    ):
        """htmx partial: next page of timeline thumbnails."""
        conn = get_conn()
        try:
            per_page = 60
            offset = page * per_page
            sel_months = _parse_months(months)

            where, params = _build_timeline_where(year, sel_months)

            rows = conn.execute(
                f"""SELECT f.id, f.path, f.content_hash, f.width, f.height,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM files f
                   LEFT JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE {where}
                   ORDER BY COALESCE(pm.taken_at, f.mtime) DESC
                   LIMIT ? OFFSET ?""",
                params + [per_page, offset],
            ).fetchall()
            total = conn.execute(
                f"""SELECT COUNT(*) FROM files f
                    LEFT JOIN photo_meta pm ON pm.file_id = f.id
                    WHERE {where}""",
                params,
            ).fetchone()[0]
            has_next = offset + per_page < total
            return TEMPLATES.TemplateResponse(request, "_photo_grid.html", {
                "photos": rows,
                "page": page,
                "has_next": has_next,
                "year": year,
                "months_qs": ",".join(sel_months),
            })
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Per-person timeline
    # ------------------------------------------------------------------
    # /person/{name}     — full page: stats + "through the years" face
    #                      strip + paginated photo grid
    # /person-page/{name} — htmx partial for infinite-scroll pagination
    #
    # "name" is URL-encoded. We aggregate across every cluster that has
    # been named the same thing, so the multi-cluster-same-person case
    # (kids aging, HDBSCAN splits) shows as a single person.

    PERSON_PER_PAGE = 60

    def _person_photos(conn, person_id: int, limit: int, offset: int):
        """All photos containing any face linked (via cluster) to this person.

        Aggregates across every cluster owned by the person, so historical
        names + multi-cluster splits collapse into one timeline.
        """
        return conn.execute(
            """SELECT f.id, f.path, f.content_hash,
                      pm.taken_at, pm.camera_model, pm.city, pm.country
               FROM (
                   SELECT DISTINCT fa.file_id AS fid
                   FROM faces fa
                   JOIN face_clusters fc ON fc.cluster_id = fa.cluster_id
                   JOIN files ff ON ff.id = fa.file_id
                   WHERE fc.person_id = ? AND ff.missing = 0
               ) t
               JOIN files f ON f.id = t.fid
               LEFT JOIN photo_meta pm ON pm.file_id = f.id
               ORDER BY COALESCE(pm.taken_at, f.mtime) DESC
               LIMIT ? OFFSET ?""",
            (person_id, limit, offset),
        ).fetchall()

    def _person_total(conn, person_id: int) -> int:
        row = conn.execute(
            """SELECT COUNT(DISTINCT fa.file_id)
               FROM faces fa
               JOIN face_clusters fc ON fc.cluster_id = fa.cluster_id
               JOIN files f ON f.id = fa.file_id
               WHERE fc.person_id = ? AND f.missing = 0""",
            (person_id,),
        ).fetchone()
        return row[0] if row else 0

    def _resolve_person_or_404(conn, name: str):
        """Look up a person by any historical name. Returns ``(person_row,
        None)`` on success or ``(None, HTMLResponse(404))`` if unknown."""
        from .. import persons as persons_mod
        pid = persons_mod.find_person_by_name(conn, name)
        if pid is None:
            return None, HTMLResponse(f"No person named '{name}'.", 404)
        p = persons_mod.get_person(conn, pid)
        if p is None:
            return None, HTMLResponse(f"No person named '{name}'.", 404)
        return p, None

    @app.get("/person/{name}", response_class=HTMLResponse)
    async def person_detail(
        request: Request,
        name: str,
        page: int = Query(0, ge=0),
    ):
        from urllib.parse import unquote
        from .. import persons as persons_mod

        spec = unquote(name)
        conn = get_conn()
        try:
            p, err = _resolve_person_or_404(conn, spec)
            if err:
                return err
            person_id = p["id"]
            primary_name = p["primary_name"]

            # Aggregate stats across every cluster owned by this person.
            stats = conn.execute(
                """SELECT COUNT(DISTINCT fa.file_id) AS n_photos,
                          COUNT(*)                    AS n_faces,
                          COUNT(DISTINCT fa.cluster_id) AS n_clusters,
                          MIN(pm.taken_at)            AS first_date,
                          MAX(pm.taken_at)            AS last_date
                   FROM faces fa
                   JOIN face_clusters fc ON fc.cluster_id = fa.cluster_id
                   JOIN files f ON f.id = fa.file_id
                   LEFT JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE fc.person_id = ? AND f.missing = 0""",
                (person_id,),
            ).fetchone()

            if not stats or not stats["n_photos"]:
                return HTMLResponse(
                    f"'{primary_name}' has no photos yet.", 404
                )

            # "Through the years": best face per calendar year.
            year_faces = conn.execute(
                """SELECT yr, face_id, file_id, content_hash, person_name
                   FROM (
                       SELECT SUBSTR(pm.taken_at, 1, 4) AS yr,
                              fa.id          AS face_id,
                              fa.file_id     AS file_id,
                              f.content_hash AS content_hash,
                              fa.person_name AS person_name,
                              ROW_NUMBER() OVER (
                                  PARTITION BY SUBSTR(pm.taken_at, 1, 4)
                                  ORDER BY fa.det_score DESC, fa.id
                              ) AS rn
                       FROM faces fa
                       JOIN face_clusters fc ON fc.cluster_id = fa.cluster_id
                       JOIN files f ON f.id = fa.file_id
                       JOIN photo_meta pm ON pm.file_id = f.id
                       WHERE fc.person_id = ? AND f.missing = 0
                         AND pm.taken_at IS NOT NULL
                         AND LENGTH(pm.taken_at) >= 4
                   )
                   WHERE rn = 1
                   ORDER BY yr""",
                (person_id,),
            ).fetchall()

            epochs = persons_mod.list_name_epochs(conn, person_id)
            photos = _person_photos(conn, person_id, PERSON_PER_PAGE, 0)
            has_next = PERSON_PER_PAGE < stats["n_photos"]

            return TEMPLATES.TemplateResponse(request, "person.html", {
                "person": primary_name,
                "person_id": person_id,
                "epochs": epochs,
                "stats": stats,
                "year_faces": year_faces,
                "photos": photos,
                "page": 0,
                "has_next": has_next,
            })
        finally:
            conn.close()

    @app.post("/person/{name}/add-name")
    async def person_add_name(name: str, request: Request):
        from urllib.parse import unquote
        from .. import persons as persons_mod
        spec = unquote(name)
        form = await request.form()
        new_name = (form.get("name") or "").strip()
        start = (form.get("start_date") or "").strip() or None
        end = (form.get("end_date") or "").strip() or None
        if not new_name:
            return RedirectResponse(f"/person/{name}", status_code=303)
        conn = get_conn()
        try:
            pid = persons_mod.find_person_by_name(conn, spec)
            if pid is None:
                return HTMLResponse(f"No person named '{spec}'.", 404)
            conn.execute("BEGIN")
            try:
                persons_mod.add_name_epoch(
                    conn, pid, new_name, start, end
                )
                # Adding an epoch may shift which name is canonical for
                # which date. Refresh the per-face cache.
                persons_mod.refresh_face_names_for_person(conn, pid)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            # The person's primary_name might still be the old value;
            # redirect to whatever name still resolves to this person.
            p = persons_mod.get_person(conn, pid)
            from urllib.parse import quote
            return RedirectResponse(
                f"/person/{quote(p['primary_name'])}", status_code=303
            )
        finally:
            conn.close()

    @app.post("/person/{name}/delete-name/{epoch_id}")
    async def person_delete_name(name: str, epoch_id: int):
        from urllib.parse import unquote, quote
        from .. import persons as persons_mod
        spec = unquote(name)
        conn = get_conn()
        try:
            pid = persons_mod.find_person_by_name(conn, spec)
            if pid is None:
                return HTMLResponse(f"No person named '{spec}'.", 404)
            conn.execute("BEGIN")
            try:
                try:
                    persons_mod.delete_name_epoch(conn, pid, epoch_id)
                except ValueError:
                    # Last epoch — refuse, the person needs at least one name.
                    conn.execute("ROLLBACK")
                    return RedirectResponse(
                        f"/person/{name}", status_code=303
                    )
                persons_mod.refresh_face_names_for_person(conn, pid)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            p = persons_mod.get_person(conn, pid)
            return RedirectResponse(
                f"/person/{quote(p['primary_name'])}", status_code=303
            )
        finally:
            conn.close()

    @app.post("/person/{name}/set-primary/{epoch_id}")
    async def person_set_primary(name: str, epoch_id: int):
        """Promote an epoch's name to be the person's primary_name (the
        fallback used when a photo has no date)."""
        from urllib.parse import unquote, quote
        from .. import persons as persons_mod
        spec = unquote(name)
        conn = get_conn()
        try:
            pid = persons_mod.find_person_by_name(conn, spec)
            if pid is None:
                return HTMLResponse(f"No person named '{spec}'.", 404)
            row = conn.execute(
                "SELECT name FROM person_names WHERE id = ? AND person_id = ?",
                (epoch_id, pid),
            ).fetchone()
            if not row:
                return RedirectResponse(f"/person/{name}", status_code=303)
            conn.execute("BEGIN")
            try:
                persons_mod.set_primary_name(conn, pid, row["name"])
                persons_mod.refresh_face_names_for_person(conn, pid)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return RedirectResponse(
                f"/person/{quote(row['name'])}", status_code=303
            )
        finally:
            conn.close()

    @app.get("/person-page/{name}", response_class=HTMLResponse)
    async def person_page_partial(
        request: Request,
        name: str,
        page: int = Query(0, ge=0),
    ):
        """htmx partial: next page of a person's photo grid."""
        from urllib.parse import unquote
        from .. import persons as persons_mod
        spec = unquote(name)
        conn = get_conn()
        try:
            pid = persons_mod.find_person_by_name(conn, spec)
            if pid is None:
                return HTMLResponse("", 404)
            offset = page * PERSON_PER_PAGE
            photos = _person_photos(conn, pid, PERSON_PER_PAGE, offset)
            total = _person_total(conn, pid)
            has_next = offset + PERSON_PER_PAGE < total
            return TEMPLATES.TemplateResponse(
                request,
                "_person_photo_grid.html",
                {
                    "photos": photos,
                    "person": spec,
                    "page": page,
                    "has_next": has_next,
                },
            )
        finally:
            conn.close()

    @app.get("/photo/{file_id}", response_class=HTMLResponse)
    async def photo_detail(
        request: Request,
        file_id: int,
        # When any of these are present, prev/next walk the search result
        # set instead of the full timeline, so you can page through a
        # search from the detail view.
        q: str = Query(""),
        label: str = Query(""),
        camera: str = Query(""),
        person: str = Query(""),
        cluster: int | None = Query(None),
        country: str = Query(""),
        k: int = Query(60, ge=1, le=500),
    ):
        conn = get_conn()
        try:
            f = conn.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
            if not f:
                return HTMLResponse("not found", 404)
            meta = conn.execute(
                "SELECT * FROM photo_meta WHERE file_id=?", (file_id,)
            ).fetchone()
            dets = conn.execute(
                "SELECT label, conf, x, y, w, h FROM detections WHERE file_id=? ORDER BY conf DESC",
                (file_id,),
            ).fetchall()
            faces = conn.execute(
                "SELECT id, cluster_id, person_name, det_score, x, y, w, h FROM faces WHERE file_id=?",
                (file_id,),
            ).fetchall()

            search_qs = _search_qs(q, label, camera, person, cluster, country, k)
            in_search = bool(search_qs)

            prev_id: int | None = None
            next_id: int | None = None
            search_position: int | None = None
            search_total: int | None = None

            if in_search:
                # Walk the same list as /search. Result order matters:
                # CLIP uses distance, others use taken_at DESC.
                results = _run_search(conn, q, label, camera, person,
                                      cluster, country, k)
                ids = [r["id"] for r in results]
                if file_id in ids:
                    idx = ids.index(file_id)
                    search_position = idx
                    search_total = len(ids)
                    if idx > 0:
                        prev_id = ids[idx - 1]
                    if idx + 1 < len(ids):
                        next_id = ids[idx + 1]
            else:
                # Default: walk the full timeline by taken_at (same order
                # as /), newer = "prev", older = "next".
                taken_at = meta["taken_at"] if meta else None
                sort_key = taken_at or str(f["mtime"])

                prev_photo = conn.execute(
                    """SELECT f.id FROM files f
                       LEFT JOIN photo_meta pm ON pm.file_id = f.id
                       WHERE f.missing = 0
                         AND COALESCE(pm.taken_at, f.mtime) > ?
                       ORDER BY COALESCE(pm.taken_at, f.mtime) ASC
                       LIMIT 1""",
                    (sort_key,),
                ).fetchone()
                next_photo = conn.execute(
                    """SELECT f.id FROM files f
                       LEFT JOIN photo_meta pm ON pm.file_id = f.id
                       WHERE f.missing = 0
                         AND COALESCE(pm.taken_at, f.mtime) < ?
                       ORDER BY COALESCE(pm.taken_at, f.mtime) DESC
                       LIMIT 1""",
                    (sort_key,),
                ).fetchone()
                prev_id = prev_photo["id"] if prev_photo else None
                next_id = next_photo["id"] if next_photo else None

            # Jump-to-timeline link: compute the year/month filter this
            # photo belongs to and which page of the filtered grid it sits
            # on. Anchor #photo-{id} scrolls the grid to it on load.
            timeline_link: str | None = None
            if meta and meta["taken_at"]:
                ta = meta["taken_at"]
                tl_year = ta[:4]
                tl_month = ta[5:7].lstrip("0") or "0"
                # Count photos newer than this one under the same filter.
                sort_key = ta
                where = ("f.missing = 0 AND pm.taken_at LIKE ? "
                         "AND SUBSTR(pm.taken_at, 6, 2) = ?")
                newer = conn.execute(
                    f"""SELECT COUNT(*) FROM files f
                        LEFT JOIN photo_meta pm ON pm.file_id = f.id
                        WHERE {where}
                          AND COALESCE(pm.taken_at, f.mtime) > ?""",
                    (f"{tl_year}%", ta[5:7], sort_key),
                ).fetchone()[0]
                per_page = 60
                tl_page = newer // per_page
                page_qs = f"&page={tl_page}" if tl_page else ""
                timeline_link = (
                    f"/?year={tl_year}&months={tl_month}{page_qs}"
                    f"#photo-{file_id}"
                )

            return TEMPLATES.TemplateResponse(request, "photo.html", {
                "file": f,
                "meta": meta,
                "detections": dets,
                "faces": faces,
                "prev_id": prev_id,
                "next_id": next_id,
                "search_qs": search_qs,
                "in_search": in_search,
                "search_position": search_position,
                "search_total": search_total,
                "timeline_link": timeline_link,
                "people": _all_people(conn),
            })
        finally:
            conn.close()

    def _run_search(conn, q: str, label: str, camera: str, person: str,
                    cluster: int | None, country: str, k: int) -> list:
        """Execute the first matching filter and return a list of result rows.

        Used by both the /search page and /photo/{id} (so prev/next on the
        detail view can walk the same result set).
        """
        if q:
            from ..models.clip import embed_text
            vec = embed_text(q)
            vec_bytes = struct.pack(f"{len(vec)}f", *vec.tolist())
            return conn.execute(
                """SELECT v.rowid AS id, v.distance, f.path, f.content_hash,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM vec_clip v
                   JOIN files f ON f.id = v.rowid
                   LEFT JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE v.embedding MATCH ? AND k = ?
                   ORDER BY v.distance""",
                (vec_bytes, k),
            ).fetchall()
        if label:
            return conn.execute(
                """SELECT DISTINCT f.id, f.path, f.content_hash,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM detections d
                   JOIN files f ON f.id = d.file_id
                   LEFT JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE d.label = ? AND f.missing = 0
                   ORDER BY pm.taken_at DESC
                   LIMIT ?""",
                (label, k),
            ).fetchall()
        if person:
            return conn.execute(
                """SELECT DISTINCT f.id, f.path, f.content_hash,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM faces fa
                   JOIN files f ON f.id = fa.file_id
                   LEFT JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE fa.person_name = ? AND f.missing = 0
                   ORDER BY pm.taken_at DESC
                   LIMIT ?""",
                (person, k),
            ).fetchall()
        if cluster is not None:
            return conn.execute(
                """SELECT DISTINCT f.id, f.path, f.content_hash,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM faces fa
                   JOIN files f ON f.id = fa.file_id
                   LEFT JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE fa.cluster_id = ? AND f.missing = 0
                   ORDER BY pm.taken_at DESC
                   LIMIT ?""",
                (cluster, k),
            ).fetchall()
        if camera:
            return conn.execute(
                """SELECT f.id, f.path, f.content_hash,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM files f
                   JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE pm.camera_model = ? AND f.missing = 0
                   ORDER BY pm.taken_at DESC
                   LIMIT ?""",
                (camera, k),
            ).fetchall()
        if country:
            return conn.execute(
                """SELECT f.id, f.path, f.content_hash,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM files f
                   JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE pm.country = ? AND f.missing = 0
                   ORDER BY pm.taken_at DESC
                   LIMIT ?""",
                (country, k),
            ).fetchall()
        return []

    def _search_qs(q: str, label: str, camera: str, person: str,
                   cluster: int | None, country: str, k: int) -> str:
        """Build a '?q=...&label=...' query-string for threading search
        filters through photo detail links. Includes the leading '?' so
        it can be appended directly to a path, or is empty when no filter
        is active. k is included so the result set size is stable."""
        from urllib.parse import urlencode
        parts: list[tuple[str, str]] = []
        if q: parts.append(("q", q))
        if label: parts.append(("label", label))
        if camera: parts.append(("camera", camera))
        if person: parts.append(("person", person))
        if cluster is not None: parts.append(("cluster", str(cluster)))
        if country: parts.append(("country", country))
        if not parts:
            return ""
        parts.append(("k", str(k)))
        return "?" + urlencode(parts)

    @app.get("/search", response_class=HTMLResponse)
    async def search_page(
        request: Request,
        q: str = Query("", description="CLIP text query"),
        label: str = Query("", description="YOLO label filter"),
        camera: str = Query("", description="Camera model filter"),
        person: str = Query("", description="Person name filter"),
        cluster: int | None = Query(None, description="Unnamed face cluster id"),
        country: str = Query("", description="Country filter"),
        k: int = Query(60, ge=1, le=500),
    ):
        conn = get_conn()
        try:
            photos = _run_search(conn, q, label, camera, person, cluster, country, k)

            # Dropdowns for search form
            labels = [r[0] for r in conn.execute(
                "SELECT DISTINCT label FROM detections ORDER BY label"
            ).fetchall()]
            cameras = [r[0] for r in conn.execute(
                "SELECT DISTINCT camera_model FROM photo_meta WHERE camera_model IS NOT NULL ORDER BY camera_model"
            ).fetchall()]
            people = [r[0] for r in conn.execute(
                "SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL ORDER BY person_name"
            ).fetchall()]
            countries = [r[0] for r in conn.execute(
                "SELECT DISTINCT country FROM photo_meta WHERE country IS NOT NULL ORDER BY country"
            ).fetchall()]

            return TEMPLATES.TemplateResponse(request, "search.html", {
                "photos": photos,
                "q": q, "label": label, "camera": camera,
                "person": person, "cluster": cluster, "country": country,
                "labels": labels, "cameras": cameras,
                "people": people, "countries": countries,
                "search_qs": _search_qs(q, label, camera, person, cluster, country, k),
            })
        finally:
            conn.close()

    FACES_PER_PAGE = 48

    def _load_face_clusters(conn, page: int) -> tuple[list, int, bool]:
        """Return (clusters_on_this_page, total_clusters, has_next).

        Live-counts faces via a LEFT JOIN so stale `face_clusters.face_count`
        values (from clusters whose source photos were later deleted) don't
        produce phantom empty cards on the grid. Empty clusters are excluded
        via HAVING.
        """
        offset = page * FACES_PER_PAGE
        # Total = number of clusters that currently have ≥1 face.
        total = conn.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM faces WHERE cluster_id IS NOT NULL"
        ).fetchone()[0]
        rows = conn.execute(
            """SELECT fc.cluster_id,
                      fc.person_name,
                      COUNT(fa.id) AS face_count,
                      (SELECT f.content_hash FROM faces fa2
                       JOIN files f ON f.id = fa2.file_id
                       WHERE fa2.cluster_id = fc.cluster_id
                       ORDER BY fa2.det_score DESC LIMIT 1) AS rep_hash,
                      (SELECT fa2.file_id FROM faces fa2
                       WHERE fa2.cluster_id = fc.cluster_id
                       ORDER BY fa2.det_score DESC LIMIT 1) AS rep_file_id
               FROM face_clusters fc
               LEFT JOIN faces fa ON fa.cluster_id = fc.cluster_id
               GROUP BY fc.cluster_id
               HAVING COUNT(fa.id) > 0
               ORDER BY face_count DESC
               LIMIT ? OFFSET ?""",
            (FACES_PER_PAGE, offset),
        ).fetchall()
        has_next = offset + FACES_PER_PAGE < total
        return rows, total, has_next

    def _all_people(conn) -> list[str]:
        """All known names (every epoch of every person) for autocomplete."""
        from .. import persons as persons_mod
        return persons_mod.all_known_names(conn)

    @app.get("/faces", response_class=HTMLResponse)
    async def faces_page(request: Request, page: int = Query(0, ge=0)):
        conn = get_conn()
        try:
            clusters, total, has_next = _load_face_clusters(conn, page)
            return TEMPLATES.TemplateResponse(request, "faces.html", {
                "clusters": clusters,
                "total": total,
                "page": page,
                "has_next": has_next,
                "people": _all_people(conn),
            })
        finally:
            conn.close()

    @app.get("/faces-page", response_class=HTMLResponse)
    async def faces_page_partial(request: Request, page: int = Query(0, ge=0)):
        """htmx partial: next page of face cluster cards."""
        conn = get_conn()
        try:
            clusters, _, has_next = _load_face_clusters(conn, page)
            return TEMPLATES.TemplateResponse(request, "_faces_grid.html", {
                "clusters": clusters,
                "page": page,
                "has_next": has_next,
            })
        finally:
            conn.close()

    def _apply_cluster_name(conn, cluster_id: int, name: str) -> int:
        """Assign ``name`` to ``cluster_id``, routing through the persons
        identity model.

        Behaviour:
        * If the typed name resolves (via any name epoch) to an existing
          person, the cluster joins that person — multiple clusters can
          belong to one identity (e.g. kids aging into a different
          HDBSCAN cluster).
        * If no person matches, a new person is created with this name as
          its sole open-ended epoch.
        * Either way, ``faces.person_name`` for every face in this cluster
          is recomputed to the date-appropriate name from the person's
          epochs.

        Returns the cluster_id (always unchanged — clusters are no longer
        merged at this layer; identity unification happens via persons).
        """
        from .. import persons as persons_mod

        person_id = persons_mod.get_or_create_person(conn, name)
        # Cache primary_name on the cluster so legacy queries still work.
        primary = persons_mod.get_person(conn, person_id)["primary_name"]
        conn.execute(
            "UPDATE face_clusters SET person_id=?, person_name=? WHERE cluster_id=?",
            (person_id, primary, cluster_id),
        )
        # Recompute per-face cached names using each face's own taken_at.
        persons_mod.refresh_face_names_for_person(conn, person_id)
        return cluster_id

    @app.get("/api/people.json")
    async def api_people():
        """JSON list of known people for autocomplete.

        Shape: `[{"name": "Alice", "count": 42}, ...]`, sorted by count desc
        so common names appear first in the datalist.
        """
        conn = get_conn()
        try:
            rows = conn.execute(
                """SELECT person_name AS name, COUNT(*) AS count
                   FROM faces
                   WHERE person_name IS NOT NULL AND person_name != ''
                   GROUP BY person_name
                   ORDER BY count DESC"""
            ).fetchall()
            return [{"name": r["name"], "count": r["count"]} for r in rows]
        finally:
            conn.close()

    @app.post("/faces/{cluster_id}/name")
    async def name_face(cluster_id: int, request: Request):
        form = await request.form()
        name = form.get("name", "").strip()
        if not name:
            return RedirectResponse("/faces", status_code=303)
        conn = get_conn()
        try:
            _apply_cluster_name(conn, cluster_id, name)
        finally:
            conn.close()
        return RedirectResponse("/faces", status_code=303)

    @app.post("/face/{face_id}/name")
    async def name_single_face(face_id: int, request: Request):
        """Name a single face and propagate to its cluster, auto-merging
        into an existing same-named cluster if one exists."""
        form = await request.form()
        name = form.get("name", "").strip()
        file_id = form.get("file_id", "")
        if not name:
            return RedirectResponse(f"/photo/{file_id}", status_code=303)
        conn = get_conn()
        try:
            # Name this specific face first (covers the unclustered case).
            conn.execute(
                "UPDATE faces SET person_name=? WHERE id=?", (name, face_id)
            )
            row = conn.execute(
                "SELECT cluster_id FROM faces WHERE id=?", (face_id,)
            ).fetchone()
            if row and row["cluster_id"] is not None:
                _apply_cluster_name(conn, row["cluster_id"], name)
        finally:
            conn.close()
        return RedirectResponse(f"/photo/{file_id}", status_code=303)

    @app.post("/faces/merge")
    async def merge_clusters(request: Request):
        """Merge multiple face clusters into one.

        Expects form fields:
        - cluster_ids: comma-separated list of cluster IDs to merge
        - name: the person_name to assign to the merged cluster

        The first cluster_id becomes the surviving cluster; all faces in
        the other clusters are reassigned to it, their face_clusters rows
        deleted, and the face count updated.
        """
        form = await request.form()
        raw_ids = form.get("cluster_ids", "")
        name = form.get("name", "").strip()

        try:
            ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
        except ValueError:
            return RedirectResponse("/faces", status_code=303)
        if len(ids) < 2 or not name:
            return RedirectResponse("/faces", status_code=303)

        keeper = ids[0]
        others = ids[1:]

        conn = get_conn()
        try:
            # Move all faces from 'others' clusters into 'keeper'
            for cid in others:
                conn.execute(
                    "UPDATE faces SET cluster_id=?, person_name=? WHERE cluster_id=?",
                    (keeper, name, cid),
                )
                conn.execute(
                    "DELETE FROM face_clusters WHERE cluster_id=?",
                    (cid,),
                )
            # Update keeper
            conn.execute(
                "UPDATE faces SET person_name=? WHERE cluster_id=?",
                (name, keeper),
            )
            new_count = conn.execute(
                "SELECT COUNT(*) FROM faces WHERE cluster_id=?",
                (keeper,),
            ).fetchone()[0]
            conn.execute(
                "UPDATE face_clusters SET person_name=?, face_count=? WHERE cluster_id=?",
                (name, new_count, keeper),
            )
        finally:
            conn.close()
        return RedirectResponse("/faces", status_code=303)

    @app.get("/cameras", response_class=HTMLResponse)
    async def cameras_page(request: Request):
        conn = get_conn()
        try:
            cameras = conn.execute(
                """SELECT pm.camera_make, pm.camera_model,
                          COUNT(*) AS cnt,
                          MIN(pm.taken_at) AS first_seen,
                          MAX(pm.taken_at) AS last_seen
                   FROM photo_meta pm
                   JOIN files f ON f.id = pm.file_id
                   WHERE pm.camera_model IS NOT NULL AND f.missing = 0
                   GROUP BY pm.camera_make, pm.camera_model
                   ORDER BY cnt DESC"""
            ).fetchall()
            return TEMPLATES.TemplateResponse(request, "cameras.html", {
                "cameras": cameras,
            })
        finally:
            conn.close()

    def _lens_options(conn, camera_model: str) -> list[dict]:
        """Distinct *physical* lenses for a camera, with friendly labels.

        For phones with multiple lens modules (iPhone "back triple
        camera 4.25mm" / "back triple camera 1.54mm" / "front camera
        2.71mm" etc.), this enables a pill row on /camera/{model} so you
        can drill into "just the ultrawide shots" or "just selfies".

        Why grouping matters: iOS reports the same physical lens with
        different lens strings depending on which camera mode the user
        was in ("back triple camera 4.25mm" vs "back dual wide camera
        4.25mm" vs "back camera 4.25mm" — all the same Main lens). We
        group by ``(orientation, focal_mm)`` so each physical lens gets
        exactly one pill that filters across all the EXIF variants.

        Labelling strategy:
        * Strip the camera model prefix that EXIF tools repeat in the
          lens string.
        * Detect "front" vs "back" by keyword.
        * For back cameras, sort groups by focal length and assign
          relative position labels (shortest → "Ultra wide", longest →
          "Telephoto", in between → "Main") so the names are stable
          across phone generations even though absolute mm differs.

        Returns a list of dicts; each ``value`` is a comma-separated
        list of raw lens strings to be matched via SQL ``IN``.
        """
        import re

        rows = conn.execute(
            """SELECT pm.lens AS lens, COUNT(*) AS cnt
               FROM photo_meta pm
               JOIN files f ON f.id = pm.file_id
               WHERE pm.camera_model = ? AND f.missing = 0
               GROUP BY pm.lens
               ORDER BY cnt DESC""",
            (camera_model,),
        ).fetchall()

        m_lower = camera_model.lower()

        def parse(lens: str | None) -> tuple[str, float | None]:
            if not lens or lens == "65535":
                return ("unknown", None)
            s = lens.lower()
            if s.startswith(m_lower):
                s = s[len(m_lower):].strip()
            orient = "front" if ("front" in s or "selfie" in s) else "back"
            fm = re.search(r"(\d+(?:\.\d+)?)\s*mm", s)
            focal = float(fm.group(1)) if fm else None
            return (orient, focal)

        # Group raw lens rows by (orient, rounded focal). Round to 0.1mm
        # to coalesce trivial floating-point variations ("4.25mm" stays
        # distinct from "6.0mm"). Lenses with no focal length get their
        # own bucket per orientation.
        groups: dict[tuple[str, float | None], dict] = {}
        for r in rows:
            orient, focal = parse(r["lens"])
            key = (orient, round(focal, 1) if focal else None)
            g = groups.setdefault(key, {
                "orient": orient,
                "focal": focal,
                "lenses": [],
                "count": 0,
            })
            if r["lens"] is not None:
                g["lenses"].append(r["lens"])
            g["count"] += r["cnt"]

        if len(groups) <= 1:
            return []  # one physical lens — no pill row needed

        # Rank back-camera groups by focal length for relative labels.
        # Goal: every pill gets a *unique* short label (the focal length
        # is also shown, but the position word is the at-a-glance hint).
        back_sorted = sorted(
            [g for k, g in groups.items() if k[0] == "back" and g["focal"]],
            key=lambda g: g["focal"],
        )
        N = len(back_sorted)
        if N == 1:
            label_seq = ["Main"]
        elif N == 2:
            label_seq = ["Wide", "Tele"]
        elif N == 3:
            label_seq = ["Ultra wide", "Main", "Telephoto"]
        elif N == 4:
            label_seq = ["Ultra wide", "Main", "Tele", "Telephoto"]
        else:
            # 5+: keep the extremes named, fall back to focal length only
            # for the middle (no position word — the focal length is the
            # disambiguator). This avoids "Main / Main / Main".
            label_seq = ["Ultra wide"] + [""] * (N - 2) + ["Telephoto"]
        position_label = {id(g): lbl for g, lbl in zip(back_sorted, label_seq)}

        out: list[dict] = []
        # Sort: back lenses first (by focal asc), then front, then unknown.
        order_key = lambda g: (
            {"back": 0, "front": 1, "unknown": 2}[g["orient"]],
            g["focal"] or 999,
        )
        for g in sorted(groups.values(), key=order_key):
            if g["orient"] == "front":
                label = "Front"
                if g["focal"]:
                    label += f" · {g['focal']:g}mm"
            elif g["orient"] == "unknown":
                label = "(no lens info)"
            else:
                pos = position_label.get(id(g))
                if pos and g["focal"]:
                    label = f"{pos} · {g['focal']:g}mm"
                elif pos:
                    label = pos
                elif g["focal"]:
                    label = f"Back · {g['focal']:g}mm"
                else:
                    label = "Back"

            # The pill's "value" is a tab-separated list of lens strings
            # (tab is safe — never appears in EXIF lens fields). The
            # detail route splits this back to an IN-clause.
            value = "\t".join(g["lenses"])  # empty string if all NULL
            tooltip = "\n".join(g["lenses"]) if g["lenses"] else "(no lens info)"
            out.append({
                "value": value,
                "label": label,
                "count": g["count"],
                "tooltip": tooltip,
            })
        return out

    CAMERA_PER_PAGE = 60

    def _camera_where(camera_model: str, lens: str | None) -> tuple[str, list]:
        """Build WHERE + params for the camera detail page.

        ``lens`` is a tab-separated bundle of EXIF lens strings that all
        map to one physical lens (iOS reports the same lens differently
        per camera mode). Empty-string param means the "(no lens info)"
        pill — matches NULL lens rows.
        """
        where = "pm.camera_model = ? AND f.missing = 0"
        params: list = [camera_model]
        if lens is not None:
            if lens == "":
                where += " AND pm.lens IS NULL"
            else:
                parts = [p for p in lens.split("\t") if p]
                if parts:
                    placeholders = ",".join("?" * len(parts))
                    where += f" AND pm.lens IN ({placeholders})"
                    params.extend(parts)
        return where, params

    def _camera_photos(conn, where, params, limit, offset):
        return conn.execute(
            f"""SELECT f.id, f.path, f.content_hash, f.width, f.height,
                       pm.taken_at, pm.camera_model, pm.city, pm.country
                FROM files f
                JOIN photo_meta pm ON pm.file_id = f.id
                WHERE {where}
                ORDER BY pm.taken_at DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

    @app.get("/camera/{camera_model}", response_class=HTMLResponse)
    async def camera_detail(
        request: Request,
        camera_model: str,
        page: int = Query(0, ge=0),
        lens: str | None = Query(None, description="Filter by exact lens string"),
    ):
        conn = get_conn()
        try:
            where, params = _camera_where(camera_model, lens)
            total = conn.execute(
                f"""SELECT COUNT(*) FROM files f
                    JOIN photo_meta pm ON pm.file_id = f.id
                    WHERE {where}""",
                params,
            ).fetchone()[0]
            # Initial render: load every page from 0..page inclusive, so the
            # rendered grid height matches what was on screen when the user
            # navigated away. The browser then restores scroll position via
            # the default history.scrollRestoration = "auto".
            load_count = (page + 1) * CAMERA_PER_PAGE
            rows = _camera_photos(conn, where, params, load_count, 0)
            has_next = load_count < total
            return TEMPLATES.TemplateResponse(request, "camera_detail.html", {
                "camera_model": camera_model,
                "photos": rows,
                "page": page,
                "has_next": has_next,
                "total": total,
                "lens_options": _lens_options(conn, camera_model),
                "selected_lens": lens,
            })
        finally:
            conn.close()

    @app.get("/camera-page/{camera_model}", response_class=HTMLResponse)
    async def camera_page_partial(
        request: Request,
        camera_model: str,
        page: int = Query(0, ge=0),
        lens: str | None = Query(None),
    ):
        """htmx partial: next page of the camera-detail photo grid."""
        conn = get_conn()
        try:
            where, params = _camera_where(camera_model, lens)
            offset = page * CAMERA_PER_PAGE
            rows = _camera_photos(conn, where, params, CAMERA_PER_PAGE, offset)
            total = conn.execute(
                f"""SELECT COUNT(*) FROM files f
                    JOIN photo_meta pm ON pm.file_id = f.id
                    WHERE {where}""",
                params,
            ).fetchone()[0]
            has_next = offset + CAMERA_PER_PAGE < total
            return TEMPLATES.TemplateResponse(
                request, "_camera_grid.html",
                {
                    "photos": rows,
                    "camera_model": camera_model,
                    "selected_lens": lens,
                    "page": page,
                    "has_next": has_next,
                },
            )
        finally:
            conn.close()

    NEARBY_PER_PAGE = 60

    def _nearby_scored(conn, lat, lon, radius_km) -> list[tuple[float, dict]]:
        """Compute haversine-filtered, distance-sorted photos within radius.

        Returned list is the full result set (small, since we bbox-prefilter
        and then exact-filter in Python). Pagination just slices it.
        """
        dlat = radius_km / 111.32
        dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
        lat_min, lat_max = lat - dlat, lat + dlat
        lon_min, lon_max = lon - dlon, lon + dlon

        rows = conn.execute(
            """SELECT f.id, f.path, f.content_hash,
                      pm.taken_at, pm.camera_model, pm.city, pm.country,
                      pm.lat, pm.lon
               FROM photo_meta pm
               JOIN files f ON f.id = pm.file_id
               WHERE pm.lat BETWEEN ? AND ?
                 AND pm.lon BETWEEN ? AND ?
                 AND f.missing = 0""",
            (lat_min, lat_max, lon_min, lon_max),
        ).fetchall()

        R = 6371.0  # km
        lat_r = math.radians(lat)
        scored: list[tuple[float, dict]] = []
        for r in rows:
            plat_r = math.radians(r["lat"])
            plon_r = math.radians(r["lon"])
            dph = plat_r - lat_r
            dlh = plon_r - math.radians(lon)
            a = math.sin(dph / 2) ** 2 + math.cos(lat_r) * math.cos(plat_r) * math.sin(dlh / 2) ** 2
            dist = 2 * R * math.asin(min(1.0, math.sqrt(a)))
            if dist <= radius_km:
                scored.append((dist, dict(r)))
        scored.sort(key=lambda x: x[0])
        return scored

    @app.get("/nearby", response_class=HTMLResponse)
    async def nearby_page(
        request: Request,
        lat: float = Query(..., description="Center latitude"),
        lon: float = Query(..., description="Center longitude"),
        radius_km: float = Query(5.0, gt=0, le=20000, description="Search radius in km"),
        page: int = Query(0, ge=0),
    ):
        """Photos within *radius_km* of (lat, lon), nearest first."""
        conn = get_conn()
        try:
            scored = _nearby_scored(conn, lat, lon, radius_km)
            total = len(scored)
            # Initial render loads pages 0..page inclusive so a back-button
            # restore brings back the same scroll height.
            load_count = (page + 1) * NEARBY_PER_PAGE
            window = scored[:load_count]
            photos = [dict(d, distance_km=dist) for dist, d in window]
            has_next = load_count < total

            return TEMPLATES.TemplateResponse(request, "nearby.html", {
                "photos": photos,
                "lat": lat,
                "lon": lon,
                "radius_km": radius_km,
                "total": total,
                "page": page,
                "has_next": has_next,
            })
        finally:
            conn.close()

    @app.get("/nearby-page", response_class=HTMLResponse)
    async def nearby_page_partial(
        request: Request,
        lat: float = Query(...),
        lon: float = Query(...),
        radius_km: float = Query(5.0, gt=0, le=20000),
        page: int = Query(0, ge=0),
    ):
        """htmx partial: next page of the nearby photo grid."""
        conn = get_conn()
        try:
            scored = _nearby_scored(conn, lat, lon, radius_km)
            total = len(scored)
            offset = page * NEARBY_PER_PAGE
            window = scored[offset:offset + NEARBY_PER_PAGE]
            photos = [dict(d, distance_km=dist) for dist, d in window]
            has_next = offset + NEARBY_PER_PAGE < total
            return TEMPLATES.TemplateResponse(
                request, "_nearby_grid.html",
                {
                    "photos": photos,
                    "lat": lat,
                    "lon": lon,
                    "radius_km": radius_km,
                    "page": page,
                    "has_next": has_next,
                },
            )
        finally:
            conn.close()

    @app.get("/map", response_class=HTMLResponse)
    async def map_page(request: Request):
        conn = get_conn()
        try:
            points = conn.execute(
                """SELECT f.id, f.content_hash, pm.lat, pm.lon, pm.city, pm.country,
                          pm.taken_at
                   FROM photo_meta pm
                   JOIN files f ON f.id = pm.file_id
                   WHERE pm.lat IS NOT NULL AND pm.lon IS NOT NULL AND f.missing = 0"""
            ).fetchall()
            return TEMPLATES.TemplateResponse(request, "map.html", {
                "points": points,
            })
        finally:
            conn.close()

    @app.get("/about", response_class=HTMLResponse)
    async def about_page(request: Request):
        conn = get_conn()
        try:
            roots = conn.execute(
                "SELECT path, last_scanned_at FROM scan_roots ORDER BY path"
            ).fetchall()

            last_index_at = db.get_meta(conn, "last_index_at")
            last_cluster_at = db.get_meta(conn, "last_cluster_at")

            total_images = conn.execute(
                "SELECT COUNT(*) FROM files WHERE missing=0"
            ).fetchone()[0]
            total_bytes = conn.execute(
                "SELECT COALESCE(SUM(size), 0) FROM files WHERE missing=0"
            ).fetchone()[0]
            detections_n = conn.execute(
                "SELECT COUNT(*) FROM detections"
            ).fetchone()[0]
            faces_n = conn.execute(
                "SELECT COUNT(*) FROM faces"
            ).fetchone()[0]
            clusters_n = conn.execute(
                "SELECT COUNT(*) FROM face_clusters"
            ).fetchone()[0]

            db_size = cfg.db_path.stat().st_size if cfg.db_path.exists() else 0
            # Include WAL + shm if present (they can be sizeable mid-write)
            for sfx in ("-wal", "-shm"):
                side = cfg.db_path.with_name(cfg.db_path.name + sfx)
                if side.exists():
                    db_size += side.stat().st_size

            # Pre-format rows so the template stays dumb
            roots_fmt = [
                {
                    "path": r["path"],
                    "last_scanned": _fmt_ts(str(r["last_scanned_at"])),
                }
                for r in roots
            ]

            return TEMPLATES.TemplateResponse(request, "about.html", {
                "roots": roots_fmt,
                "last_index_at": _fmt_ts(last_index_at),
                "last_cluster_at": _fmt_ts(last_cluster_at),
                "total_images": total_images,
                "total_bytes_human": _fmt_bytes(total_bytes),
                "detections_n": detections_n,
                "faces_n": faces_n,
                "clusters_n": clusters_n,
                "db_size_human": _fmt_bytes(db_size),
                "db_path": str(cfg.db_path),
            })
        finally:
            conn.close()

    # Browsers (especially iOS Safari) hit these root paths regardless of
    # any <link rel="icon"> tag. Serve them directly from /static so we
    # don't return 404s for every page load from a fresh client.
    @app.get("/favicon.ico")
    async def serve_favicon():
        return FileResponse(WEB_DIR / "static" / "favicon.ico",
                            media_type="image/x-icon")

    @app.get("/apple-touch-icon.png")
    @app.get("/apple-touch-icon-precomposed.png")
    async def serve_apple_touch_icon():
        return FileResponse(WEB_DIR / "static" / "apple-touch-icon.png",
                            media_type="image/png")

    @app.get("/thumb/{content_hash}")
    async def serve_thumb(content_hash: str):
        """Serve a 512px thumbnail, generating + caching on demand.

        Older photos (and photos from machines that haven't run
        `index --thumbs` yet) may have no cached thumb. Rather than
        404'ing — which produces the broken-image-with-alt-text mess on
        grid pages — we look up the source file by content_hash, decode
        it, write the thumb to the cache, then serve it. Subsequent
        requests hit the cache.
        """
        p = thumb_path(cfg.cache_dir, content_hash)
        if p.exists():
            return FileResponse(p, media_type="image/jpeg")

        conn = get_conn()
        try:
            row = conn.execute(
                """SELECT path FROM files
                   WHERE content_hash = ? AND missing = 0
                   LIMIT 1""",
                (content_hash,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return HTMLResponse("not found", 404)

        src = Path(row["path"])
        if not src.exists():
            return HTMLResponse("source file missing", 404)

        try:
            from .. import decode, thumbs as thumbs_mod
            arr = decode.load_image(src)
            from ..thumbs import ensure_thumbnail
            out = ensure_thumbnail(arr, cfg.cache_dir, content_hash)
            return FileResponse(out, media_type="image/jpeg")
        except Exception as e:
            log.warning("thumb generation failed for %s: %s", src, e)
            return HTMLResponse("decode error", 500)

    # Formats browsers can render natively in <img>
    _WEB_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".avif"})

    @app.get("/full/{file_id}")
    async def serve_full(file_id: int):
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT path, content_hash FROM files WHERE id=?", (file_id,)
            ).fetchone()
            if not row:
                return HTMLResponse("not found", 404)
            p = Path(row["path"])
            # For formats browsers can't display (HEIC, RAW, TIFF, ...),
            # serve the pre-generated JPEG thumbnail instead.
            if p.suffix.lower() not in _WEB_EXTS:
                tp = thumb_path(cfg.cache_dir, row["content_hash"])
                if tp.exists():
                    return FileResponse(tp, media_type="image/jpeg")
            if not p.exists():
                return HTMLResponse("file missing", 404)
            return FileResponse(p)
        finally:
            conn.close()

    return app


# ---- CLI registration ----

def register(parent: typer.Typer) -> None:
    @parent.command(name="serve")
    def cmd_serve(
        port: int = typer.Option(8765, "--port", "-p", help="Port for the web UI."),
        host: str = typer.Option(
            "0.0.0.0", "--host",
            help="Interface to bind. Defaults to all interfaces (0.0.0.0). "
                 "Use 127.0.0.1 to restrict to localhost.",
        ),
        reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
    ) -> None:
        """Start the web UI server."""
        import uvicorn
        cfg = config.load()
        db.init(cfg.db_path)

        # Detailed log config: timestamps on every line, and filter out the
        # context-free "Invalid HTTP request received" warnings that come from
        # bots / TLS-on-HTTP probes (they carry no useful info since the
        # request was unparseable).
        log_config: dict = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelprefix)s %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                    "()": "uvicorn.logging.DefaultFormatter",
                },
                "access": {
                    "format": '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                    "()": "uvicorn.logging.AccessFormatter",
                },
            },
            "filters": {
                "no_invalid_http": {
                    "()": "imagewizard.web.log_filter.InvalidHTTPFilter",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "access": {
                    "formatter": "access",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": "INFO",
                    "filters": ["no_invalid_http"],
                },
                "uvicorn.access": {
                    "handlers": ["access"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }

        if reload:
            # uvicorn reload needs an import string, not an app object
            import os
            os.environ.setdefault("IMAGEWIZARD_DATA_DIR", str(cfg.data_dir))
            os.environ.setdefault("IMAGEWIZARD_CACHE_DIR", str(cfg.cache_dir))
            uvicorn.run(
                "imagewizard.web.app:create_app",
                factory=True,
                host=host,
                port=port,
                reload=True,
                reload_dirs=[str(Path(__file__).parent.parent)],
                log_config=log_config,
            )
        else:
            app = create_app(cfg)
            uvicorn.run(app, host=host, port=port, log_config=log_config)
