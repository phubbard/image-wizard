"""FastAPI web application for browsing the photo index."""

from __future__ import annotations

import datetime as _dt
import math
import struct
from pathlib import Path


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
        """Return (clusters_on_this_page, total_clusters, has_next)."""
        offset = page * FACES_PER_PAGE
        total = conn.execute("SELECT COUNT(*) FROM face_clusters").fetchone()[0]
        rows = conn.execute(
            """SELECT fc.cluster_id, fc.person_name, fc.face_count,
                      (SELECT f.content_hash FROM faces fa
                       JOIN files f ON f.id = fa.file_id
                       WHERE fa.cluster_id = fc.cluster_id
                       ORDER BY fa.det_score DESC LIMIT 1) AS rep_hash,
                      (SELECT fa.file_id FROM faces fa
                       WHERE fa.cluster_id = fc.cluster_id
                       ORDER BY fa.det_score DESC LIMIT 1) AS rep_file_id
               FROM face_clusters fc
               ORDER BY fc.face_count DESC
               LIMIT ? OFFSET ?""",
            (FACES_PER_PAGE, offset),
        ).fetchall()
        has_next = offset + FACES_PER_PAGE < total
        return rows, total, has_next

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

    @app.post("/faces/{cluster_id}/name")
    async def name_face(cluster_id: int, request: Request):
        form = await request.form()
        name = form.get("name", "").strip()
        if not name:
            return RedirectResponse("/faces", status_code=303)
        conn = get_conn()
        try:
            conn.execute(
                "UPDATE face_clusters SET person_name=? WHERE cluster_id=?",
                (name, cluster_id),
            )
            conn.execute(
                "UPDATE faces SET person_name=? WHERE cluster_id=?",
                (name, cluster_id),
            )
        finally:
            conn.close()
        return RedirectResponse("/faces", status_code=303)

    @app.post("/face/{face_id}/name")
    async def name_single_face(face_id: int, request: Request):
        """Name a single face and propagate to its cluster."""
        form = await request.form()
        name = form.get("name", "").strip()
        file_id = form.get("file_id", "")
        if not name:
            return RedirectResponse(f"/photo/{file_id}", status_code=303)
        conn = get_conn()
        try:
            # Update this face
            conn.execute(
                "UPDATE faces SET person_name=? WHERE id=?", (name, face_id)
            )
            # If it belongs to a cluster, propagate name to the whole cluster
            row = conn.execute(
                "SELECT cluster_id FROM faces WHERE id=?", (face_id,)
            ).fetchone()
            if row and row["cluster_id"] is not None:
                cid = row["cluster_id"]
                conn.execute(
                    "UPDATE faces SET person_name=? WHERE cluster_id=?",
                    (name, cid),
                )
                conn.execute(
                    "UPDATE face_clusters SET person_name=? WHERE cluster_id=?",
                    (name, cid),
                )
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

    @app.get("/camera/{camera_model}", response_class=HTMLResponse)
    async def camera_detail(request: Request, camera_model: str, page: int = Query(0, ge=0)):
        conn = get_conn()
        try:
            per_page = 60
            offset = page * per_page
            total = conn.execute(
                """SELECT COUNT(*) FROM files f
                   JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE pm.camera_model = ? AND f.missing = 0""",
                (camera_model,),
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT f.id, f.path, f.content_hash, f.width, f.height,
                          pm.taken_at, pm.camera_model, pm.city, pm.country
                   FROM files f
                   JOIN photo_meta pm ON pm.file_id = f.id
                   WHERE pm.camera_model = ? AND f.missing = 0
                   ORDER BY pm.taken_at DESC
                   LIMIT ? OFFSET ?""",
                (camera_model, per_page, offset),
            ).fetchall()
            has_next = offset + per_page < total
            return TEMPLATES.TemplateResponse(request, "camera_detail.html", {
                "camera_model": camera_model,
                "photos": rows,
                "page": page,
                "has_next": has_next,
                "total": total,
            })
        finally:
            conn.close()

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
            per_page = 60
            # Bounding-box pre-filter so the index is used; haversine refines.
            # 1 deg lat ≈ 111.32 km; 1 deg lon ≈ 111.32*cos(lat) km.
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

            # Exact haversine; filter + sort in Python (small result set).
            R = 6371.0  # km
            lat_r = math.radians(lat)
            lon_r = math.radians(lon)
            scored: list[tuple[float, dict]] = []
            for r in rows:
                plat_r = math.radians(r["lat"])
                plon_r = math.radians(r["lon"])
                dph = plat_r - lat_r
                dlh = plon_r - lon_r
                a = math.sin(dph / 2) ** 2 + math.cos(lat_r) * math.cos(plat_r) * math.sin(dlh / 2) ** 2
                dist = 2 * R * math.asin(min(1.0, math.sqrt(a)))
                if dist <= radius_km:
                    scored.append((dist, dict(r)))
            scored.sort(key=lambda x: x[0])

            total = len(scored)
            offset = page * per_page
            window = scored[offset:offset + per_page]
            photos = [dict(d, distance_km=dist) for dist, d in window]
            has_next = offset + per_page < total

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

    @app.get("/map", response_class=HTMLResponse)
    async def map_page(request: Request):
        conn = get_conn()
        try:
            points = conn.execute(
                """SELECT f.id, f.content_hash, pm.lat, pm.lon, pm.city, pm.country,
                          pm.taken_at
                   FROM photo_meta pm
                   JOIN files f ON f.id = pm.file_id
                   WHERE pm.lat IS NOT NULL AND pm.lon IS NOT NULL AND f.missing = 0
                   LIMIT 5000"""
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
        p = thumb_path(cfg.cache_dir, content_hash)
        if p.exists():
            return FileResponse(p, media_type="image/jpeg")
        return HTMLResponse("not found", 404)

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
            )
        else:
            app = create_app(cfg)
            uvicorn.run(app, host=host, port=port)
