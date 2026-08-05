"""HTTP 路由：/api/v1/history/* —— 07 会话历史记录的查询/检索/导出接口。

与 app/routes/chat.py 同范式：本文件只做参数解析与调用编排，业务逻辑全部在
app.services.history（store / search / export）内。user_id 同时作为作用域与鉴权
标识（与 SessionHistoryStore 的 requester_id 对齐）；管理员的越权查看由 store 的
admin_user_ids 控制，路由层不重复实现。
"""
from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Query

from app.services.history.store import get_history_store
from app.services.history.models import SearchHit, SessionDetail, SessionMeta

router = APIRouter(tags=["history"])


@router.get("/v1/history/sessions", response_model=List[SessionMeta])
async def list_sessions(
    user_id: str = Query(..., description="请求者/归属用户ID，作为作用域与鉴权标识"),
    scene: Optional[str] = Query(None, description="按场景过滤（qa/analyze/...）"),
    status: str = Query("active", description="active | archived | all"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[SessionMeta]:
    store = get_history_store()
    return await store.list_sessions(
        user_id, user_id=user_id, scene=scene, status=status, limit=limit, offset=offset
    )


@router.get("/v1/history/sessions/{conversation_id}", response_model=Optional[SessionDetail])
async def get_session(
    conversation_id: str,
    user_id: str = Query(..., description="请求者ID，作为鉴权标识"),
) -> Optional[SessionDetail]:
    store = get_history_store()
    return await store.get_session(user_id, conversation_id)


@router.get("/v1/history/search", response_model=List[SearchHit])
async def search_history(
    user_id: str = Query(..., description="请求者ID，作为作用域与鉴权标识"),
    q: str = Query(..., description="检索关键词", min_length=1),
    scene: Optional[str] = Query(None, description="按场景过滤"),
    limit: int = Query(20, ge=1, le=100),
) -> List[SearchHit]:
    store = get_history_store()
    return await store.keyword_search(user_id, q, scene=scene, limit=limit)


@router.get("/v1/history/sessions/{conversation_id}/export")
async def export_session(
    conversation_id: str,
    user_id: str = Query(..., description="请求者ID，作为鉴权标识"),
    fmt: Literal["md", "json"] = Query("md", description="导出格式"),
) -> dict:
    store = get_history_store()
    content = await store.export_session(user_id, conversation_id, fmt=fmt)
    return {"conversation_id": conversation_id, "fmt": fmt, "content": content}
