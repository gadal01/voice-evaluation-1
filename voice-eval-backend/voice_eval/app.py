"""The evaluation backend as a FastAPI app.   Run:  python -m voice_eval     (default http://127.0.0.1:8001)

API under /api (X-API-Key), live OpenAPI at /openapi.json and docs at /docs. If UI_DIST points at the built
voice-eval-ui, it is served at / so one process is enough for a demo.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from voice_eval.config import get_settings
from voice_eval.router import router


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="Voice agent evaluation API")
    if s.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in s.cors_origins.split(",") if o.strip()],
                           allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])
    app.include_router(router)

    if s.ui_dist:
        dist = Path(s.ui_dist).resolve()

        @app.get("/", include_in_schema=False)
        async def index() -> Response:
            f = dist / "index.html"
            if not f.exists():
                return Response(f"UI_DIST={dist} has no index.html: build voice-eval-ui first (npm run build)", status_code=503, media_type="text/plain")
            return FileResponse(f)

        @app.get("/assets/{name:path}", include_in_schema=False)
        async def asset(name: str) -> FileResponse:
            root = (dist / "assets").resolve()
            f = (root / name).resolve()   # resolved and checked so a request can never leave dist/assets
            if root not in f.parents or not f.is_file():
                raise HTTPException(404)
            return FileResponse(f)

    return app


app = create_app()
