"""HTTP 路由：/api/v1/history/* —— 07 会话历史记录的查询/检索/导出接口。

与 app/routes/chat.py 同范式：本文件只做参数解析与调用编排，业务逻辑全部在
app.db.history（store / search / export）内。``user_id`` 作用域由 ``CurrentUser``
（JWT 解析）提供（09 数据隔离），路由层不再接收查询参数 ``user_id``。
管理员的越权查看由 store 的 admin_user_ids 控制，路由层不重复实现。
"""
from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter

from app.db.history.models import SearchHit, SessionDetail, SessionMeta
from app.db.history.store import get_history_store
from app.db.models import User
from app.routes.deps import CurrentUser

router = APIRouter(tags=["history"])


@router.get("/v1/history/sessions", response_model=List[SessionMeta])
async def list_sessions(
    current_user: User = CurrentUser,
    scene: Literal["qa", "analyze", None] = None,
    status: str = "active",
    limit: int = 50,
    offset: int = 0,
) -> List[SessionMeta]:
    store = get_history_store()
    return await store.list_sessions(
        current_user.id, user_id=str(current_user.id), scene=scene, status=status, limit=limit, offset=offset
    )


@router.get("/v1/history/sessions/{conversation_id}", response_model=SessionDetail | None)
async def get_session(
    conversation_id: str, current_user: User = CurrentUser
) -> SessionDetail | None:
    store = get_history_store()
    return await store.get_session(str(current_user.id), conversation_id)


@router.get("/v1/history/search", response_model=List[SearchHit])
async def search_history(
    current_user: User = CurrentUser,
    q: str = None,
    scene: Literal["qa", "analyze", None] = None,
    limit: int = 20,
) -> List[SearchHit]:
    store = get_history_store()
    return await store.keyword_search(str(current_user.id), q, scene=scene, limit=limit)


@router.get("/v1/history/sessions/{conversation_id}/export")
async def export_session(
    conversation_id: str,
    current_user: User = CurrentUser,
    fmt: Literal["md", "json"] = "md",
) -> dict:
    store = get_history_store()
    content = await store.export_session(str(current_user.id), conversation_id, fmt=fmt)
    return {"conversation_id": conversation_id, "fmt": fmt, "content": content}
