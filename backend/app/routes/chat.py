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
from app.services.chat_service import chat as chat_service

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
    return await chat_service(req, db, user_id=current_user.id)


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    req: AnalyzeRequest,
    db: Optional[AsyncSession] = Depends(get_db),
) -> AnalyzeResponse:
    return await analyze_service(req)
