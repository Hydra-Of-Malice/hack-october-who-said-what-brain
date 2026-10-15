"""The FastAPI application: REST, SSE and the live audio socket (docs/API_CONTRACT.md), plus the built web
UI from apps/web/dist when it exists."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..brain.llm import LLMError
from ..brand import REPO_ROOT, brand
from ..config import settings
from ..graph import GraphError, GraphUnavailable
from . import ask, rest, stream
from .access import Access, AccessSettings
from .services import DISCONNECT_GRACE_S, Services

log = logging.getLogger(__name__)

WEB_DIST = REPO_ROOT / "apps" / "web" / "dist"
_API_PREFIXES = ("api", "ws")


def create_app(
    *, disconnect_grace_s: float = DISCONNECT_GRACE_S, web_dist: Path | None = WEB_DIST
) -> FastAPI:
    """Build the app. Settings are read now (CORS, access) and in the lifespan (everything else)."""
    _configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = await Services.start(disconnect_grace_s=disconnect_grace_s)
        app.state.services = services
        try:
            yield
        finally:
            await services.shutdown()

    app = FastAPI(
        title=brand()["name"],
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    origins = _cors_origins()
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    access_cfg = AccessSettings()
    hosts = [h.strip() for h in access_cfg.allowed_hosts.split(",") if h.strip()]
    if hosts:  # stops DNS rebinding of a server on 127.0.0.1
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    app.state.access = Access(access_cfg, origins)
    _add_error_handlers(app)
    app.include_router(rest.router)
    app.include_router(ask.router)
    app.include_router(stream.router)
    if web_dist is not None and (web_dist / "index.html").is_file():
        _mount_web(app, web_dist)
    return app


def _cors_origins() -> list[str]:
    """Other sites allowed to call the API from a browser. The UI is served by this app (or proxied by the
    Vite dev server), so it needs none: only origins named in WSW_CORS_ORIGINS are allowed, and the settings
    default "*" applies only when it is set explicitly (a wildcard lets any page a visitor opens read the
    transcripts of a server on their machine)."""
    cfg = settings()
    if "cors_origins" not in cfg.model_fields_set:
        return []
    return [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]


def _add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = []
        for error in exc.errors():
            where = ".".join(
                str(p) for p in error.get("loc", ()) if p not in ("body", "query", "path", "form")
            )
            problems.append(f"{where}: {error.get('msg', 'invalid')}" if where else str(error.get("msg")))
        return JSONResponse(status_code=422, content={"detail": "; ".join(problems) or "invalid request"})

    @app.exception_handler(GraphUnavailable)
    async def graph_unavailable(_: Request, exc: GraphUnavailable) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": f"The graph database is unavailable: {exc}"})

    @app.exception_handler(GraphError)
    async def graph_error(_: Request, exc: GraphError) -> JSONResponse:
        log.error("graph query failed: %s", exc)
        return JSONResponse(status_code=500, content={"detail": f"Graph query failed: {exc}"})

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": f"The Gemma 4 runtime failed: {exc}"})


def _mount_web(app: FastAPI, dist: Path) -> None:
    """Serve the built UI: hashed assets as static files, anything else that is not /api or /ws falls back
    to index.html so client-side routes (/m/:id, /commitments) survive a reload."""
    root = dist.resolve()
    assets = root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    index = root / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    async def web(path: str) -> Response:
        if path.split("/", 1)[0] in _API_PREFIXES:
            return JSONResponse(status_code=404, content={"detail": f"no such endpoint: /{path}"})
        candidate = (root / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("wsw_brain").setLevel(logging.INFO)
