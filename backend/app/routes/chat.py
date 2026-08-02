"""HTTP 路由：/health、/v1/chat、/v1/analyze（传统 REST，均 async）。"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    get_db,
)
from app.services import analyze as analyze_service
from app.services import chat as chat_service

router = APIRouter(tags=["agent"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    db: Optional[AsyncSession] = Depends(get_db),
) -> ChatResponse:
    return await chat_service(req, db)


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    req: AnalyzeRequest,
    db: Optional[AsyncSession] = Depends(get_db),
) -> AnalyzeResponse:
    return await analyze_service(req)
