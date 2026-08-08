"""HTTP 路由：/api/health、/api/v1/chat、/api/v1/chat/stream、/api/v1/analyze。"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.system import run_stream
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
    req.user_id = current_user.id
    return await chat_service_module.chat(req, db, user_id=current_user.id)


@router.post("/v1/chat/stream")
async def chat_stream_endpoint(
    req: ChatRequest,
    current_user: User = CurrentUser,
) -> StreamingResponse:
    """SSE 流式多智能体对话：每个专家回复逐条推送到前端。"""
    req.user_id = current_user.id
    use_personal = bool(getattr(req, "use_personal_docs", True))

    async def event_stream():
        try:
            async for ev in run_stream(
                message=req.message,
                scene=req.scene,
                user_id=current_user.id,
                use_personal_docs=use_personal,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    req: AnalyzeRequest,
    db: Optional[AsyncSession] = Depends(get_db),
) -> AnalyzeResponse:
    return await analyze_service(req)
