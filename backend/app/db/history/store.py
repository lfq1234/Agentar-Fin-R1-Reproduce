"""07-会话历史记录：SQLite 存储实现 + Noop 兜底 + 权限仲裁 + 降级。

设计（单表收口，评审：避免会话记录过度拆分）：
- 03 的 ``conversations`` 表即「会话记录单表」：一次会话一行（id 主键），
  会话头（user_id/scene/title/时间）作列，聊天正文 + 多专家执行轨迹自包含于
  ``data``(JSON) 字段
  ``{"messages":[{role,content,scene,created_at,            -- 用户/助手消息
                  "trace":{run_id,turn_id,created_at,model,status,duration_ms,
                           total_tokens,final_result,events:[事件树]}}]}`  -- 仅助手消息带 trace
  原 messages / session_meta / session_traces / trace_events / history_embeddings 表已废弃并删除，
  全部会话记录收口于 ``conversations.data``（单表，无历史拆分表）。
- 权限仲裁（B2/B3）：所有读/删方法收 ``requester_id``，集中做「本人 或 管理员」。
- conversations 时间字段为 ISO TEXT（UTC），本模块内部比较/排序统一转 epoch 毫秒。
- 写入前对所有文本字段统一脱敏（redact，S2）。
- fail_mode=silent 时写失败仅告警，绝不阻断主链路（S8）。
- init_tables 幂等建表 + 幂等迁移（B6）。

锁模型：对外公开方法各自获取一次 ``asyncio.Lock``，内部调用无锁原始助手
（_execute/_fetchall/_fetchone），避免 asyncio.Lock 不可重入死锁。
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from app.db.history.models import (
    HistoryAccessError,
    SearchHit,
    SessionDetail,
    SessionMeta,
    TraceEvent,
    TurnDetail,
)
from app.db.history.redact import redact_text, redact_value

logger = logging.getLogger("app.db.history")

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema/sqlite/main.sql"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _to_epoch_ms(value: Any) -> Optional[int]:
    """把 int/float/ISO 字符串归一化为 epoch 毫秒；无法解析返回 None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value > 1e11:
            return int(value)
        return int(value * 1000)
    if isinstance(value, str):
        s = value.strip()
        if s.lstrip("-").isdigit():
            return _to_epoch_ms(int(s))
        iso = s.replace("Z", "+00:00")
        try:
            dt = datetime.datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def _epoch_ms_to_iso(ms: int) -> str:
    """epoch 毫秒 → ISO UTC 字符串（用于 conversations.updated_at 的 < 比较，ISO 字典序即时间序）。"""
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).isoformat()


def _as_dict(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    try:
        return asdict(result)
    except Exception:
        return {"reply": getattr(result, "reply", str(result))}


def _truncate_title(text: str, limit: int = 50) -> Optional[str]:
    if not text:
        return None
    t = text.strip().replace("\n", " ")
    return t[:limit]


def _event_to_flat_dict(ev: TraceEvent) -> dict:
    """TraceEvent → 平铺 dict（含 parent_event_id 与空 children，供 _nest_dict_events 嵌套）。"""
    meta = ev.meta if isinstance(ev.meta, dict) else {}
    return {
        "id": ev.id,
        "parent_event_id": ev.parent_event_id,
        "agent": ev.agent or "",
        "type": ev.type or "",
        "summary_in": ev.summary_in or "",
        "summary_out": ev.summary_out or "",
        "meta_json": json.dumps(meta, ensure_ascii=False) if meta else "",
        "tokens": ev.tokens or 0,
        "duration_ms": ev.duration_ms or 0,
        "created_at": ev.created_at or 0,
        "children": [],
    }


def _nest_dict_events(flat: list[dict]) -> list[dict]:
    """按 parent_event_id 把平铺事件嵌套成树（无父/父缺失 → 根）。"""
    by_id: dict = {}
    roots: list[dict] = []
    for d in flat:
        by_id[d["id"]] = d
    for d in flat:
        pid = d.get("parent_event_id")
        if pid is not None and pid in by_id:
            by_id[pid]["children"].append(d)
        else:
            roots.append(d)
    return roots


def _flatten_events(nodes: list[dict]):
    """把嵌套事件树拍平为 TraceEvent 列表（供 TurnDetail.events）。"""
    for n in nodes or []:
        yield _dict_to_event(n)
        yield from _flatten_events(n.get("children", []))


def _dict_to_event(d: dict) -> TraceEvent:
    meta_json = d.get("meta_json")
    if not meta_json and d.get("meta"):
        meta_json = json.dumps(d["meta"], ensure_ascii=False)
    return TraceEvent(
        id=d.get("id"),
        run_id=d.get("run_id"),
        parent_event_id=d.get("parent_event_id"),
        turn_id=d.get("turn_id"),
        seq=d.get("seq", 0),
        agent=d.get("agent", ""),
        type=d.get("type", ""),
        summary_in=d.get("summary_in", ""),
        summary_out=d.get("summary_out", ""),
        meta_json=meta_json or "",
        tokens=d.get("tokens", 0),
        duration_ms=d.get("duration_ms", 0),
        created_at=d.get("created_at", 0),
    )


def _row_to_meta(row: Any) -> SessionMeta:
    """conversations 一行 → SessionMeta（从 data 推导 msg_count/total_tokens/时间）。"""
    data = {}
    raw = row["data"] if "data" in row.keys() else None
    if raw:
        try:
            data = json.loads(raw) or {}
        except Exception:
            data = {}
    msgs = data.get("messages", [])
    msg_count = len(msgs)
    total_tokens = sum(
        (m.get("trace") or {}).get("total_tokens", 0) for m in msgs if m.get("trace")
    )
    title = row["title"]
    if not title:
        for m in msgs:
            if m.get("role") == "user" and m.get("content"):
                title = _truncate_title(m["content"])
                break
    times = [m.get("created_at") for m in msgs if m.get("created_at")]
    first = min(times) if times else (_to_epoch_ms(row["created_at"]) or 0)
    last = max(times) if times else (_to_epoch_ms(row["updated_at"]) or 0)
    return SessionMeta(
        conversation_id=str(row["id"]),
        scene=row["scene"],
        title=title,
        status="active",
        msg_count=msg_count,
        total_tokens=total_tokens,
        first_at=first,
        last_at=last,
        created_at=_to_epoch_ms(row["created_at"]) or 0,
    )


# ---------------------------------------------------------------------------
# Noop 实现（history.enabled=false 兜底，主链路零阻塞）
# ---------------------------------------------------------------------------
class NoopHistoryStore:
    enabled = False

    async def connect(self) -> None:
        return None

    async def init_tables(self) -> None:
        return None

    async def record_run(self, **_kwargs: Any) -> None:
        return None

    async def list_sessions(self, *args: Any, **_kwargs: Any) -> list[SessionMeta]:
        return []

    async def get_session(self, *args: Any, **_kwargs: Any) -> Optional[SessionDetail]:
        return None

    async def get_turn(self, *args: Any, **_kwargs: Any) -> Optional[TurnDetail]:
        return None

    async def keyword_search(self, *args: Any, **_kwargs: Any) -> list[SearchHit]:
        return []

    async def semantic_search(self, *args: Any, **_kwargs: Any) -> list[SearchHit]:
        return []

    async def export_session(self, *args: Any, **_kwargs: Any) -> str:
        return ""

    async def get_session_for_review(self, *args: Any, **_kwargs: Any) -> str:
        return ""

    async def delete_session(self, *args: Any, **_kwargs: Any) -> int:
        return 0

    async def purge_before(self, *args: Any, **_kwargs: Any) -> int:
        return 0


# ---------------------------------------------------------------------------
# 原始（无锁）DB 助手
# ---------------------------------------------------------------------------
class _RawDB:
    """把无锁的原始执行助手挂到 SessionHistoryStore 上，避免二次加锁死锁。"""

    async def _execute(self, sql: str, params: list) -> Any:
        await self._ensure()
        return await self._conn.execute(sql, params)

    async def _fetchall(self, sql: str, params: list) -> list:
        await self._ensure()
        cur = await self._conn.execute(sql, params)
        return await cur.fetchall()

    async def _fetchone(self, sql: str, params: list):
        await self._ensure()
        cur = await self._conn.execute(sql, params)
        return await cur.fetchone()


# ---------------------------------------------------------------------------
# SQLite 实现（单表 conversations）
# ---------------------------------------------------------------------------
class SessionHistoryStore(_RawDB):
    enabled = True

    def __init__(
        self,
        db_path: str,
        *,
        record_traces: bool = True,
        async_write: bool = True,
        fail_mode: str = "silent",  # silent | local
        pii_redact: bool = True,
        semantic_search: bool = False,
        allow_admin_all: bool = False,
        admin_user_ids: Optional[list] = None,
        retention_days: int = 180,
        model_name: str = "unknown",
        history_dir: Optional[str] = None,
    ) -> None:
        self.db_path = db_path
        self.record_traces = record_traces
        self.async_write = async_write
        self.fail_mode = fail_mode
        self.pii_redact = pii_redact
        self.semantic_search = semantic_search
        self.allow_admin_all = allow_admin_all
        self._admin_user_ids = {str(x) for x in (admin_user_ids or [])}
        self.retention_days = retention_days
        self.model_name = model_name
        self.history_dir = history_dir or os.path.dirname(os.path.abspath(db_path))
        self._fallback_path = os.path.join(self.history_dir, "history_fallback.jsonl")

        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    # —— 生命周期 —— #
    async def connect(self) -> None:
        if self._conn is not None:
            return
        os.makedirs(self.history_dir, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def init_tables(self) -> None:
        """幂等建表（评审 B6）。"""
        await self._ensure()
        async with self._lock:
            sql = _SCHEMA_PATH.read_text(encoding="utf-8")
            await self._conn.executescript(sql)
            await self._conn.commit()

    async def _ensure(self) -> None:
        if self._conn is None:
            await self.connect()

    # 旧表聚合迁移已移除：单表设计，会话记录全部收口于 conversations.data，无历史拆分表需聚合。

    # —— 权限仲裁（B2/B3） —— #
    def _is_admin(self, requester_id: Any) -> bool:
        if not self.allow_admin_all:
            return False
        return str(requester_id) in self._admin_user_ids

    async def _require_access(self, conversation_id: str, requester_id: Any) -> None:
        """校验 requester 是否可访问该 conversation（本人 或 管理员）。"""
        requester_id = str(requester_id)
        row = await self._fetchone(
            "SELECT user_id FROM conversations WHERE id=?", (conversation_id,)
        )
        if row is None:
            raise HistoryAccessError(f"会话 {conversation_id} 不存在")
        if requester_id == str(row["user_id"]) or self._is_admin(requester_id):
            return
        raise HistoryAccessError(f"拒绝访问会话 {conversation_id}：非本人且非管理员")

    async def _on_write_failure(self, exc: Exception, payload: dict) -> None:
        logger.warning("07 历史写入失败（已降级，不影响主链路）: %s | payload=%s", exc, payload)
        if self.fail_mode == "local":
            try:
                os.makedirs(self.history_dir, exist_ok=True)
                with open(self._fallback_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"error": str(exc), "payload": payload, "at": _now_ms()},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception as e2:  # pragma: no cover - 兜底中的兜底
                logger.warning("07 本地兜底写入也失败: %s", e2)

    # —— 写入：聊天消息由 chat_service 写入 data；本方法把 trace 挂到该轮助手消息 —— #
    async def record_run(
        self,
        *,
        conversation_id: str,
        user_id: str,
        scene: Optional[str],
        run_id: str,
        turn_id: str,
        duration_ms: int,
        model: str,
        result: Any,
        events: list[TraceEvent],
        user_message: str = "",
        total_tokens: int = 0,
        status: str = "ok",
    ) -> None:
        if not self.enabled:
            return
        conversation_id = str(conversation_id)
        user_id = str(user_id)
        try:
            await self._ensure()
            created_at = _now_ms()
            final_dict = _as_dict(result)
            if self.pii_redact:
                final_dict = redact_value(final_dict)

            # 事件 → 平铺 dict → 脱敏 → 嵌套树（record_traces=false 时仅记头，events 留空）
            if self.record_traces and events:
                flat = [_event_to_flat_dict(ev) for ev in events]
                if self.pii_redact:
                    for d in flat:
                        d["summary_in"] = redact_text(d["summary_in"])
                        d["summary_out"] = redact_text(d["summary_out"])
                        d["meta_json"] = redact_text(d["meta_json"])
                nested = _nest_dict_events(flat)
            else:
                nested = []

            trace = {
                "run_id": run_id,
                "turn_id": turn_id,
                "created_at": created_at,
                "model": model or self.model_name,
                "status": status,
                "duration_ms": duration_ms,
                "total_tokens": total_tokens,
                "final_result": final_dict,
                "events": nested,
            }

            async with self._lock:
                row = await self._fetchone(
                    "SELECT * FROM conversations WHERE id=?", (conversation_id,)
                )
                if row is None:
                    # 兜底：chat 未落库时也自洽创建会话行（含空消息 + 带 trace 的助手消息）
                    now_iso = _now_iso()
                    await self._conn.execute(
                        "INSERT INTO conversations (id,user_id,scene,title,created_at,updated_at,data) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (
                            conversation_id,
                            user_id,
                            scene,
                            _truncate_title(user_message),
                            now_iso,
                            now_iso,
                            json.dumps({"messages": []}, ensure_ascii=False),
                        ),
                    )
                    row = await self._fetchone(
                        "SELECT * FROM conversations WHERE id=?", (conversation_id,)
                    )
                data = json.loads(row["data"] or "{}")
                msgs = data.setdefault("messages", [])
                # 把 trace 挂到该轮助手消息上：先按回复内容精确匹配，否则取最后一个无 trace 的助手消息
                target = None
                reply = final_dict.get("reply") or ""
                for m in reversed(msgs):
                    if m.get("role") == "assistant" and m.get("content") == reply:
                        target = m
                        break
                if target is None:
                    for m in reversed(msgs):
                        if m.get("role") == "assistant" and not m.get("trace"):
                            target = m
                            break
                if target is not None:
                    target["trace"] = trace
                else:
                    msgs.append(
                        {
                            "role": "assistant",
                            "content": reply,
                            "scene": scene,
                            "created_at": created_at,
                            "trace": trace,
                        }
                    )
                await self._conn.execute(
                    "UPDATE conversations SET data=?, updated_at=? WHERE id=?",
                    (json.dumps(data, ensure_ascii=False), _now_iso(), conversation_id),
                )
                await self._conn.commit()
        except Exception as exc:  # 安全网：绝不向上抛，避免阻断主链路
            await self._on_write_failure(exc, {"conversation_id": conversation_id, "run_id": run_id})

    # —— 查询/回放 —— #
    async def list_sessions(
        self,
        requester_id: str,
        *,
        user_id: Optional[str] = None,
        scene: Optional[str] = None,
        status: str = "active",
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionMeta]:
        if not self.enabled:
            return []
        requester_id = str(requester_id)
        admin = self._is_admin(requester_id)
        scope = str(user_id) if user_id is not None else requester_id
        if not admin and scope != requester_id:
            return []

        sql = "SELECT * FROM conversations"
        conds: list[str] = []
        params: list = []
        if not admin:
            conds.append("CAST(user_id AS TEXT) = ?")
            params.append(requester_id)
        else:
            if user_id is not None:
                conds.append("CAST(user_id AS TEXT) = ?")
                params.append(scope)
        if scene:
            conds.append("scene = ?")
            params.append(scene)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        async with self._lock:
            rows = await self._fetchall(sql, params)
        return [_row_to_meta(r) for r in rows]

    async def get_session(
        self,
        requester_id: str,
        conversation_id: str,
        *,
        with_trace: bool = True,
    ) -> Optional[SessionDetail]:
        if not self.enabled:
            return None
        conversation_id = str(conversation_id)
        async with self._lock:
            row = await self._fetchone(
                "SELECT * FROM conversations WHERE id=?", (conversation_id,)
            )
            if row is None:
                return None
            await self._require_access(conversation_id, requester_id)

            meta = _row_to_meta(row)
            data = json.loads(row["data"] or "{}")
            msgs = data.get("messages", [])
            messages = [
                {
                    "role": m.get("role"),
                    "content": m.get("content"),
                    "scene": m.get("scene"),
                    "created_at": m.get("created_at", 0),
                    # 02 多人对话：补全智能体身份字段，
                    # 与 chat_service._persist 写入字段对齐，避免历史加载时丢失头像/名称。
                    "agent": m.get("agent"),
                    "avatar": m.get("avatar"),
                    "mention": m.get("mention"),
                    "type": m.get("type"),
                }
                for m in msgs
            ]
            trace = [m["trace"] for m in msgs if m.get("trace")] if with_trace else None
            has_trace = bool(trace)
        return SessionDetail(
            conversation_id=conversation_id,
            meta=meta,
            messages=messages,
            trace=trace,
            has_trace=has_trace,
        )

    async def get_turn(
        self,
        requester_id: str,
        conversation_id: str,
        turn_id: str,
    ) -> Optional[TurnDetail]:
        if not self.enabled:
            return None
        conversation_id = str(conversation_id)
        async with self._lock:
            row = await self._fetchone(
                "SELECT * FROM conversations WHERE id=?", (conversation_id,)
            )
            if row is None:
                return None
            await self._require_access(conversation_id, requester_id)
            data = json.loads(row["data"] or "{}")
            for m in data.get("messages", []):
                tr = m.get("trace")
                if tr and tr.get("turn_id") == turn_id:
                    flat = list(_flatten_events(tr.get("events", [])))
                    user_msg = next((e.summary_in for e in flat if e.type == "user"), "")
                    assistant_reply = next(
                        (e.summary_out for e in flat if e.type == "assistant"),
                        (tr.get("final_result") or {}).get("reply", ""),
                    )
                    return TurnDetail(
                        conversation_id=conversation_id,
                        turn_id=turn_id,
                        user_message=user_msg,
                        assistant_reply=assistant_reply,
                        events=flat,
                        final_result=tr.get("final_result"),
                    )
        return None

    # —— 检索 —— #
    async def keyword_search(
        self,
        requester_id: str,
        query: str,
        *,
        scene: Optional[str] = None,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        limit: int = 20,
    ) -> list[SearchHit]:
        from app.db.history.search import keyword_search as _kw

        async with self._lock:
            return await _kw(self, requester_id, query, scene=scene, start=start, end=end, limit=limit)

    async def semantic_search(
        self,
        requester_id: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchHit]:
        from app.db.history.search import semantic_search as _sem

        async with self._lock:
            return await _sem(self, requester_id, query, top_k=top_k)

    # —— 导出/复盘（委托 export.py） —— #
    async def export_session(
        self,
        requester_id: str,
        conversation_id: str,
        fmt: str = "md",
    ) -> str:
        from app.db.history.export import render_export

        detail = await self.get_session(requester_id, conversation_id, with_trace=True)
        if detail is None:
            return ""
        return render_export(detail, fmt=fmt)

    async def get_session_for_review(
        self,
        requester_id: str,
        conversation_id: str,
    ) -> str:
        from app.db.history.export import render_review_context

        detail = await self.get_session(requester_id, conversation_id, with_trace=True)
        if detail is None:
            return ""
        return render_review_context(detail)

    # —— 删除/清理 —— #
    async def delete_session(
        self,
        requester_id: str,
        conversation_id: str,
    ) -> int:
        if not self.enabled:
            return 0
        conversation_id = str(conversation_id)
        await self._require_access(conversation_id, requester_id)
        try:
            async with self._lock:
                await self._conn.execute(
                    "DELETE FROM conversations WHERE id=?", (conversation_id,)
                )
                await self._conn.commit()
            return 1
        except Exception as exc:
            await self._on_write_failure(exc, {"conversation_id": conversation_id})
            return 0

    async def purge_before(
        self,
        requester_id: str,
        before_date: Any,
        *,
        cascade: bool = False,
    ) -> int:
        """清理 before_date 之前的会话（按 updated_at）。仅管理员可调用（评审 B2）。"""
        if not self.enabled:
            return 0
        if not self._is_admin(requester_id):
            raise HistoryAccessError("purge_before 仅管理员可调用")
        before = _to_epoch_ms(before_date)
        if before is None:
            return 0
        before_iso = _epoch_ms_to_iso(before)
        try:
            async with self._lock:
                await self._conn.execute(
                    "DELETE FROM conversations WHERE updated_at < ? OR updated_at IS NULL",
                    (before_iso,),
                )
                await self._conn.commit()
            return 1
        except Exception as exc:
            await self._on_write_failure(exc, {"before_date": str(before_date)})
            return 0


# ---------------------------------------------------------------------------
# 工厂：单例 store + 建表入口（供 lifespan / 测试调用）
# ---------------------------------------------------------------------------
_store: Optional["SessionHistoryStore"] = None


def _resolve_db_path() -> str:
    """解析主库路径：优先 history.db_path 覆盖，否则复用 03 的 db.sqlite_path。"""
    from app.config import config

    hist = config.get("history") or {}
    override = hist.get("db_path")
    if override:
        p = Path(override)
    else:
        db = config.get("db") or {}
        p = Path(db.get("sqlite_path") or "./agentar.db")
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[3] / p  # backend/
    return str(p)


def _build_store() -> Any:
    from app.config import config

    hist = config.get("history") or {}
    if not hist.get("enabled", True):
        return NoopHistoryStore()
    return SessionHistoryStore(
        db_path=_resolve_db_path(),
        record_traces=hist.get("record_traces", True),
        async_write=hist.get("async_write", True),
        fail_mode=hist.get("fail_mode", "silent"),
        pii_redact=hist.get("pii_redact", True),
        semantic_search=hist.get("semantic_search", False),
        allow_admin_all=hist.get("allow_admin_all", False),
        admin_user_ids=hist.get("admin_user_ids") or [],
        retention_days=hist.get("retention_days", 180),
        model_name=hist.get("model_name") or "unknown",
    )


def get_history_store() -> Any:
    """返回 07 历史存储单例（按需惰性构建）。"""
    global _store
    if _store is None:
        _store = _build_store()
    return _store


def reset_history_store() -> None:
    """测试/重载时清空单例（下次 get 重建）。"""
    global _store
    _store = None


async def init_history_db() -> None:
    """启动时幂等建表 + 迁移旧表（评审 B6）。"""
    store = get_history_store()
    await store.connect()
    await store.init_tables()
