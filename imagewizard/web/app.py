"""FastAPI web application for browsing the photo index."""

from __future__ import annotations

import struct
from pathlib import Path

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

    @app.get("/", response_class=HTMLResponse)
    async def timeline(request: Request, page: int = Query(0, ge=0), year: str = Query("", description="Filter by year")):
        conn = get_conn()
        try:
            per_page = 60

            # Available years for nav bar
            years = [r[0] for r in conn.execute(
                """SELECT DISTINCT SUBSTR(taken_at, 1, 4) AS yr
                   FROM photo_meta WHERE taken_at IS NOT NULL
                   ORDER BY yr DESC"""
            ).fetchall()]

            # Build WHERE clause
            where = "f.missing = 0"
            params: list = []
            if year:
                where += " AND pm.taken_at LIKE ?"
                params.append(f"{year}%")

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
            })
        finally:
            conn.close()

    @app.get("/timeline-page", response_class=HTMLResponse)
    async def timeline_page(request: Request, page: int = Query(0, ge=0), year: str = Query("")):
        """htmx partial: next page of timeline thumbnails."""
        conn = get_conn()
        try:
            per_page = 60
            offset = page * per_page

            where = "f.missing = 0"
            params: list = []
            if year:
                where += " AND pm.taken_at LIKE ?"
                params.append(f"{year}%")

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
            })
        finally:
            conn.close()

    @app.get("/photo/{file_id}", response_class=HTMLResponse)
    async def photo_detail(request: Request, file_id: int):
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

            # Prev/next navigation (by date order, same as timeline)
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

            return TEMPLATES.TemplateResponse(request, "photo.html", {
                "file": f,
                "meta": meta,
                "detections": dets,
                "faces": faces,
                "prev_id": prev_photo["id"] if prev_photo else None,
                "next_id": next_photo["id"] if next_photo else None,
            })
        finally:
            conn.close()

    @app.get("/search", response_class=HTMLResponse)
    async def search_page(
        request: Request,
        q: str = Query("", description="CLIP text query"),
        label: str = Query("", description="YOLO label filter"),
        camera: str = Query("", description="Camera model filter"),
        person: str = Query("", description="Person name filter"),
        country: str = Query("", description="Country filter"),
        k: int = Query(60, ge=1, le=500),
    ):
        conn = get_conn()
        try:
            photos = []

            if q:
                from ..models.clip import embed_text
                vec = embed_text(q)
                vec_bytes = struct.pack(f"{len(vec)}f", *vec.tolist())
                photos = conn.execute(
                    """SELECT v.rowid AS id, v.distance, f.path, f.content_hash,
                              pm.taken_at, pm.camera_model, pm.city, pm.country
                       FROM vec_clip v
                       JOIN files f ON f.id = v.rowid
                       LEFT JOIN photo_meta pm ON pm.file_id = f.id
                       WHERE v.embedding MATCH ? AND k = ?
                       ORDER BY v.distance""",
                    (vec_bytes, k),
                ).fetchall()
            elif label:
                photos = conn.execute(
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
            elif person:
                photos = conn.execute(
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
            elif camera:
                photos = conn.execute(
                    """SELECT f.id, f.path, f.content_hash,
                              pm.taken_at, pm.camera_model, pm.city, pm.country
                       FROM files f
                       JOIN photo_meta pm ON pm.file_id = f.id
                       WHERE pm.camera_model = ? AND f.missing = 0
                       ORDER BY pm.taken_at DESC
                       LIMIT ?""",
                    (camera, k),
                ).fetchall()
            elif country:
                photos = conn.execute(
                    """SELECT f.id, f.path, f.content_hash,
                              pm.taken_at, pm.camera_model, pm.city, pm.country
                       FROM files f
                       JOIN photo_meta pm ON pm.file_id = f.id
                       WHERE pm.country = ? AND f.missing = 0
                       ORDER BY pm.taken_at DESC
                       LIMIT ?""",
                    (country, k),
                ).fetchall()

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
                "person": person, "country": country,
                "labels": labels, "cameras": cameras,
                "people": people, "countries": countries,
            })
        finally:
            conn.close()

    @app.get("/faces", response_class=HTMLResponse)
    async def faces_page(request: Request):
        conn = get_conn()
        try:
            clusters = conn.execute(
                """SELECT fc.cluster_id, fc.person_name, fc.face_count,
                          (SELECT f.content_hash FROM faces fa
                           JOIN files f ON f.id = fa.file_id
                           WHERE fa.cluster_id = fc.cluster_id
                           ORDER BY fa.det_score DESC LIMIT 1) AS rep_hash,
                          (SELECT fa.file_id FROM faces fa
                           WHERE fa.cluster_id = fc.cluster_id
                           ORDER BY fa.det_score DESC LIMIT 1) AS rep_file_id
                   FROM face_clusters fc
                   ORDER BY fc.face_count DESC"""
            ).fetchall()
            return TEMPLATES.TemplateResponse(request, "faces.html", {
                "clusters": clusters,
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
