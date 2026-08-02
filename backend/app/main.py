"""FastAPI 入口：lifespan + CORS(配置) + 挂载路由。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import config
from app.db.models import init_db
from app.routes import chat as chat_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


def _cors_settings() -> Dict[str, Any]:
    cors: Dict[str, Any] = (config.get("server", {}) or {}).get("cors", {}) or {}
    return {
        "allow_origins": cors.get("origins", ["*"]),
        "allow_methods": cors.get("methods", ["*"]),
        "allow_headers": cors.get("headers", ["*"]),
    }


app = FastAPI(title="Agentar-Fin-R1 Reproduce API", version="0.2.0")

_cors = _cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors["allow_origins"],
    allow_methods=_cors["allow_methods"],
    allow_headers=_cors["allow_headers"],
)

app.include_router(chat_routes.router)
