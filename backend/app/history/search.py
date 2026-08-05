"""07-历史检索：关键词检索（LIKE / 可选 FTS5）与可选语义检索（委托 06 embedder）。

实现放在独立模块，由 ``store.SessionHistoryStore`` 的对应方法委托调用（保持职责分离）。
- keyword_search：在 trace_events.summary_in/out/meta_json 与 session_traces.final_result 上
  做 LIKE 匹配，带 user_id/scene/时间过滤（时间基于 INTEGER epoch 毫秒，评审 B4）。
- semantic_search：默认关；开启时复用 06 的 get_embedder() 向量化，07 自有 history_embeddings
  做 Python 余弦召回（评审 S5：07 拥有 embeddings，不引入 DuckDB ATTACH 分叉）。
"""
from __future__ import annotations

import json
from typing import Any, Optional

from app.history.models import HistoryAccessError, SearchHit
from app.history.store import _to_epoch_ms


async def keyword_search(
    store,
    requester_id: str,
    query: str,
    *,
    scene: Optional[str] = None,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    limit: int = 20,
) -> list[SearchHit]:
    if not store.enabled or not query:
        return []
    requester_id = str(requester_id)
    admin = store._is_admin(requester_id)
    like = f"%{query}%"
    sql = (
        "SELECT te.id, te.run_id, te.turn_id, te.type, te.summary_in, te.summary_out, "
        "te.meta_json, te.created_at, st.conversation_id, st.user_id "
        "FROM trace_events te JOIN session_traces st ON te.run_id = st.run_id "
        "WHERE (te.summary_in LIKE ? OR te.summary_out LIKE ? OR te.meta_json LIKE ? "
        "OR st.final_result LIKE ?)"
    )
    params: list = [like, like, like, like]
    if not admin:
        sql += " AND st.user_id = ?"
        params.append(requester_id)
    if scene:
        sql += " AND st.scene = ?"
        params.append(scene)
    s_ms = _to_epoch_ms(start)
    e_ms = _to_epoch_ms(end)
    if s_ms is not None:
        sql += " AND st.created_at >= ?"
        params.append(s_ms)
    if e_ms is not None:
        sql += " AND st.created_at <= ?"
        params.append(e_ms)
    sql += " ORDER BY st.created_at DESC LIMIT ?"
    params.append(limit)

    rows = await store._fetchall(sql, params)
    hits: list[SearchHit] = []
    for r in rows:
        snippet_parts = [r["summary_in"] or "", r["summary_out"] or "", r["meta_json"] or ""]
        snippet = " / ".join(p for p in snippet_parts if p)
        idx = snippet.find(query)
        if idx != -1:
            snippet = snippet[max(0, idx - 30) : idx + len(query) + 30]
        hits.append(
            SearchHit(
                conversation_id=r["conversation_id"],
                turn_id=r["turn_id"],
                event_id=r["id"],
                snippet=snippet,
                score=1.0,
                type=r["type"],
                created_at=r["created_at"],
            )
        )
    return hits


async def semantic_search(
    store,
    requester_id: str,
    query: str,
    *,
    top_k: int = 5,
) -> list[SearchHit]:
    if not store.enabled or not store.semantic_search:
        return []
    requester_id = str(requester_id)
    admin = store._is_admin(requester_id)
    rows = await store._fetchall("SELECT * FROM history_embeddings", [])
    if not rows:
        return []
    try:
        from app.kb import get_knowledge_store

        embedder = get_knowledge_store()._get_embedder()
        qvec = embedder.embed([query])[0]
    except Exception as exc:
        store.logger.warning("07 语义检索向量化失败，降级为空: %s", exc)
        return []

    scored: list[tuple[float, Any]] = []
    for r in rows:
        try:
            vec = json.loads(r["embedding"])
        except Exception:
            continue
        sim = store._cosine(qvec, vec)
        scored.append((sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits: list[SearchHit] = []
    for sim, r in scored[:top_k]:
        if not admin:
            owner = await store._fetchone(
                "SELECT DISTINCT user_id FROM session_traces WHERE conversation_id=?",
                (r["conversation_id"],),
            )
            if owner is None or owner["user_id"] != requester_id:
                continue
        hits.append(
            SearchHit(
                conversation_id=r["conversation_id"],
                turn_id="",
                event_id=None,
                snippet=f"[embedding:{r['model']}]",
                score=round(sim, 4),
                type="semantic",
                created_at=0,
            )
        )
    return hits
