from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.personas import router as personas_router


def _parse_cors() -> list[str]:
    s = os.getenv("CORS_ALLOWED_ORIGINS", "")
    return [x.strip() for x in s.split(",") if x.strip()]


def create_app() -> FastAPI:
    app = FastAPI(title="SeeAct Personas API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors() or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(personas_router, prefix="/v1/personas")
    return app


app = create_app()

