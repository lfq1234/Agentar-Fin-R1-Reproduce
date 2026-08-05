"""FastAPI 入口：lifespan + CORS(配置) + 挂载路由。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import config
from app.db.models import init_db
from app.db.history import init_history_db, install_history_tracing
from app.model.exceptions import ModelInvokeError
from app.routes import chat as chat_routes
from app.routes import history as history_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # 07-会话历史记录：幂等建表 + 无侵入包裹 chat_service.chat
    await init_history_db()
    install_history_tracing()
    yield


def _cors_settings() -> Dict[str, Any]:
    cors: Dict[str, Any] = (config.get("server", {}) or {}).get("cors", {}) or {}
    return {
        "allow_origins": cors.get("origins", ["*"]),
        "allow_methods": cors.get("methods", ["*"]),
        "allow_headers": cors.get("headers", ["*"]),
    }


app = FastAPI(
    title="Agentar-Fin-R1 Reproduce API",
    version="0.2.0",
    lifespan=lifespan,
)

_cors = _cors_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors["allow_origins"],
    allow_methods=_cors["allow_methods"],
    allow_headers=_cors["allow_headers"],
)

# 统一 /api 前缀：路由变为 /api/health、/api/v1/chat、/api/v1/analyze（前后端一致）。
app.include_router(chat_routes.router, prefix="/api")
app.include_router(history_routes.router, prefix="/api")


@app.exception_handler(ModelInvokeError)
async def _model_invoke_error_handler(
    request: Request, exc: ModelInvokeError
) -> JSONResponse:
    """模型不可达/失败时统一返回 500，不向调用方泄露原始栈（评审 N2：上抛但不静默）。

    联调用例 test_degradation_when_model_down 依赖此行为断言 5xx + 错误文案。
    """
    return JSONResponse(
        status_code=500,
        content={"error": "model_invoke_failed", "detail": str(exc)},
    )
