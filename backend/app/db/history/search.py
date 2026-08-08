"""07-检索：关键词检索（在 conversations.data 上 LIKE）+ 语义检索（未启用）。

语义检索依赖 history_embeddings（原表已废弃、单表收口后不再维护），
故 semantic_search 直接降级返回空列表。
"""
from __future__ import annotations

import json
from typing import Any, Optional

from app.db.history.models import SearchHit


async def keyword_search(
    store: Any,
    requester_id: str,
    query: str,
    *,
    scene: Optional[str] = None,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    limit: int = 20,
) -> list[SearchHit]:
    """在 conversations.data(JSON) 的消息正文中做关键词匹配，按时间倒序返回命中。"""
    requester_id = str(requester_id)
    admin = store._is_admin(requester_id)

    sql = "SELECT * FROM conversations"
    conds: list[str] = []
    params: list = []
    if not admin:
        conds.append("CAST(user_id AS TEXT) = ?")
        params.append(requester_id)
    if scene:
        conds.append("scene = ?")
        params.append(scene)
    if conds:
        sql += " WHERE " + " AND ".join(conds)

    rows = await store._fetchall(sql, params)
    q = (query or "").lower()
    hits: list[SearchHit] = []
    for r in rows:
        try:
            data = json.loads(r["data"] or "{}") or {}
        except Exception:
            continue
        for idx, m in enumerate(data.get("messages", [])):
            content = m.get("content") or ""
            if q and q not in content.lower():
                continue
            # 时间过滤（epoch 毫秒）
            ca = m.get("created_at") or 0
            if start is not None and ca < (start if start > 1e11 else start * 1000):
                continue
            if end is not None and ca > (end if end > 1e11 else end * 1000):
                continue
            turn_id = (m.get("trace") or {}).get("turn_id", "") if m.get("trace") else ""
            hits.append(
                SearchHit(
                    conversation_id=str(r["id"]),
                    turn_id=turn_id,
                    event_id=idx,
                    snippet=content[:200],
                    score=1.0,
                    type=m.get("role", ""),
                    created_at=ca,
                )
            )

    hits.sort(key=lambda h: h.created_at, reverse=True)
    return hits[:limit]


async def semantic_search(
    store: Any,
    requester_id: str,
    query: str,
    *,
    top_k: int = 5,
) -> list[SearchHit]:
    """语义检索：history_embeddings 已随单表收口废弃，直接降级返回空。"""
    return []
