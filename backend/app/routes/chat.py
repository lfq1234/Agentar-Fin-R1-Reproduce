"""HTTP 路由：/api/health、/api/v1/chat、/api/v1/analyze（传统 REST，均 async）。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    User,
    get_db,
)
from app.routes.deps import CurrentUser, SessionDep
from app.services.analyze_service import analyze as analyze_service
# 通过模块对象引用 chat（而非直接 import 函数对象），
# 以便 install_history_tracing() 在启动时把 chat_service.chat 替换为带历史采集的包裹版本后，
# 本路由能动态取到包裹版本（否则拿到的是 import 时的原函数，旁路 trace 永远不触发）。
from app.services import chat_service as chat_service_module

router = APIRouter(tags=["agent"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    current_user: User = CurrentUser,
    db: Optional[AsyncSession] = SessionDep,
) -> ChatResponse:
    # 09：user_id 由鉴权令牌解析（不再信任请求体 user_id）。
    # 同步回填 req.user_id，供 07 history 旁路钩子（record_run）识别归属（ChatRequest 约定）。
    req.user_id = current_user.id
    # 经模块属性动态调用，确保命中 07 历史采集包裹版本（traced_chat）
    return await chat_service_module.chat(req, db, user_id=current_user.id)


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    req: AnalyzeRequest,
    db: Optional[AsyncSession] = Depends(get_db),
) -> AnalyzeResponse:
    return await analyze_service(req)
