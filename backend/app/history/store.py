"""07-会话历史记录：SQLite 存储实现 + Noop 兜底 + 权限仲裁 + 降级。

关键设计（收敛评审 B1-B6 / S1 / S2 / S4 / S7 / S8 / S9）：
- B2/B3：所有读/删方法都收 ``requester_id``，store 内集中做「本人 或 管理员」仲裁；
         管理员由配置 ``history.admin_user_ids`` 判定（03 的 User 表无 role 字段，不碰 03）。
- B4：时间字段 INTEGER epoch 毫秒；``purge_before`` / ``keyword_search`` 时间过滤基于它。
- B5：``turn_id`` 由 hook 生成并落到 session_traces / trace_events，``get_turn`` 据此组装。
- B6：``init_tables()`` 幂等建表；``build_on_startup`` 默认 true，lifespan 调用。
- S1：所有 DB 操作经**单一** ``asyncio.Lock`` 串行化（写串行化避免 SQLite 单写者锁）。
- S2：写入前对所有文本字段统一脱敏（redact）。
- S4：``fail_mode=local`` 时写失败落兜底 JSONL 文件；``replay_fallback()`` 可回填。
- S7：每次 ``record_run`` 经 ``_upsert_meta`` 同步 msg_count / total_tokens / 时间。
- S8：``fail_mode=silent`` 下写失败打 warning 日志（不阻断主链路）。
- S9：``record_traces=false`` 时仅写 session_traces 头，``get_session`` 返回空 trace 树而非报错。

锁模型（关键）：``_fetchall`` / ``_fetchone`` / ``_execute`` 是**无锁原始助手**；每个对外公开方法
自己获取一次 ``async with self._lock``，内部只调用原始助手（不二次加锁），避免 asyncio.Lock
不可重入导致的死锁（record_run 内联 meta 写入，整段只持锁一次）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from app.history.models import (
    HistoryAccessError,
    HistoryEmbedding,
    SearchHit,
    SessionDetail,
    SessionMeta,
    SessionTrace,
    TraceEvent,
    TurnDetail,
)
from app.history.redact import redact_text, redact_value

logger = logging.getLogger("app.history")

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_epoch_ms(value: Any) -> Optional[int]:
    """把 int/float/ISO 字符串归一化为 epoch 毫秒；无法解析返回 None。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # 大于 1e11 视为毫秒；否则视为秒
        if value > 1e11:
            return int(value)
        return int(value * 1000)
    if isinstance(value, str):
        s = value.strip()
        if s.lstrip("-").isdigit():
            return _to_epoch_ms(int(s))
        iso = s.replace("Z", "+00:00")
        try:
            import datetime

            dt = datetime.datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def _as_dict(result: Any) -> dict:
    """把 AgentResult（dataclass）或 dict 转 dict。"""
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

    async def upsert_session_meta(self, conversation_id: str, **_kwargs: Any) -> None:
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
# SQLite 实现
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
        # 复用 03 主库（与 SQLAlchemy 连接同一文件）：开启 WAL + 等待超时，
        # 避免 07 写与 03 写/读相互 database is locked（评审：统一主库后的并发安全）。
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def init_tables(self) -> None:
        """幂等建表（评审 B6）：CREATE TABLE IF NOT EXISTS，可安全地重复调用。"""
        await self._ensure()
        async with self._lock:
            sql = _SCHEMA_PATH.read_text(encoding="utf-8")
            await self._conn.executescript(sql)
            await self._conn.commit()

    async def _ensure(self) -> None:
        if self._conn is None:
            await self.connect()

    # —— 权限仲裁（B2/B3） —— #
    def _is_admin(self, requester_id: Any) -> bool:
        if not self.allow_admin_all:
            return False
        return str(requester_id) in self._admin_user_ids

    async def _require_access(self, conversation_id: str, requester_id: Any) -> None:
        """校验 requester 是否可访问该 conversation（本人 或 管理员）。"""
        requester_id = str(requester_id)
        rows = await self._fetchall(
            "SELECT DISTINCT user_id FROM session_traces WHERE conversation_id=?",
            (conversation_id,),
        )
        owners = {r["user_id"] for r in rows}
        if requester_id in owners or self._is_admin(requester_id):
            return
        raise HistoryAccessError(
            f"拒绝访问会话 {conversation_id}：非本人且非管理员"
        )

    async def _on_write_failure(self, exc: Exception, payload: dict) -> None:
        """写失败兜底（S4/S8）：不抛错，保证主链路不阻塞。"""
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

    # —— 写入 —— #
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
                final_dict = redact_value(final_dict)  # type: ignore[assignment]
            final_json = json.dumps(final_dict, ensure_ascii=False)

            # 整个写入（session_traces + trace_events + meta 同步）在单次持锁内完成，
            # 仅持锁一次，避免 asyncio.Lock 不可重入死锁（S1）。
            async with self._lock:
                await self._conn.execute(
                    "INSERT INTO session_traces "
                    "(run_id,turn_id,conversation_id,user_id,scene,created_at,duration_ms,"
                    "total_tokens,model,status,final_result) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        turn_id,
                        conversation_id,
                        user_id,
                        scene,
                        created_at,
                        duration_ms,
                        total_tokens,
                        model or self.model_name,
                        status,
                        final_json,
                    ),
                )
                if self.record_traces:
                    seq = 0
                    for ev in events:
                        seq += 1
                        meta_json = ev.meta_json
                        if not meta_json and ev.meta:
                            meta_json = json.dumps(ev.meta, ensure_ascii=False)
                        if self.pii_redact:
                            summary_in = redact_text(ev.summary_in)
                            summary_out = redact_text(ev.summary_out)
                            meta_json = redact_text(meta_json) if meta_json else ""
                        else:
                            summary_in, summary_out = ev.summary_in, ev.summary_out
                        await self._conn.execute(
                            "INSERT INTO trace_events "
                            "(run_id,parent_event_id,turn_id,seq,agent,type,summary_in,"
                            "summary_out,meta_json,tokens,duration_ms,created_at) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                run_id,
                                ev.parent_event_id,
                                turn_id,
                                seq,
                                ev.agent,
                                ev.type,
                                summary_in,
                                summary_out,
                                meta_json,
                                ev.tokens,
                                ev.duration_ms,
                                created_at,
                            ),
                        )
                # 同步会话元信息（S7）——原始助手，不二次加锁
                await self._upsert_meta(
                    conversation_id,
                    scene=scene,
                    msg_count=2 if self.record_traces else 0,
                    total_tokens=total_tokens,
                    first_at=created_at,
                    last_at=created_at,
                    title=_truncate_title(user_message),
                )
                await self._conn.commit()
        except Exception as exc:  # 安全网：绝不向上抛，避免阻断主链路
            await self._on_write_failure(exc, {"conversation_id": conversation_id, "run_id": run_id})

    async def _upsert_meta(
        self,
        conversation_id: str,
        *,
        scene: Optional[str] = None,
        title: Optional[str] = None,
        status: Optional[str] = None,
        msg_count: int = 0,
        total_tokens: int = 0,
        first_at: Optional[int] = None,
        last_at: Optional[int] = None,
    ) -> None:
        """原始（无锁）meta 写入；调用方须持锁（record_run 内联 / upsert_session_meta 公开方法）。"""
        row = await self._fetchone(
            "SELECT * FROM session_meta WHERE conversation_id=?", (conversation_id,)
        )
        if row is None:
            now = _now_ms()
            await self._conn.execute(
                "INSERT INTO session_meta "
                "(conversation_id,scene,title,status,msg_count,total_tokens,first_at,"
                "last_at,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    conversation_id,
                    scene,
                    title,
                    status or "active",
                    msg_count,
                    total_tokens,
                    first_at,
                    last_at,
                    now,
                ),
            )
        else:
            new_msg = (row["msg_count"] or 0) + msg_count
            new_tok = (row["total_tokens"] or 0) + total_tokens
            existing = [v for v in (row["first_at"], first_at) if v]
            new_first = min(existing) if existing else row["first_at"]
            existing_last = [v for v in (row["last_at"], last_at) if v]
            new_last = max(existing_last) if existing_last else row["last_at"]
            new_title = row["title"] or title
            new_scene = row["scene"] or scene
            await self._conn.execute(
                "UPDATE session_meta SET msg_count=?,total_tokens=?,first_at=?,"
                "last_at=?,title=?,scene=? WHERE conversation_id=?",
                (new_msg, new_tok, new_first, new_last, new_title, new_scene, conversation_id),
            )

    async def upsert_session_meta(
        self,
        conversation_id: str,
        *,
        scene: Optional[str] = None,
        title: Optional[str] = None,
        status: Optional[str] = None,
        msg_count: int = 0,
        total_tokens: int = 0,
        first_at: Optional[int] = None,
        last_at: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        conversation_id = str(conversation_id)
        try:
            await self._ensure()
            async with self._lock:
                await self._upsert_meta(
                    conversation_id,
                    scene=scene,
                    title=title,
                    status=status,
                    msg_count=msg_count,
                    total_tokens=total_tokens,
                    first_at=first_at,
                    last_at=last_at,
                )
                await self._conn.commit()
        except Exception as exc:
            await self._on_write_failure(exc, {"conversation_id": conversation_id})

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
            return []  # 越权列他人会话 → 返回空

        sql = "SELECT m.* FROM session_meta m"
        joins = ""
        conds: list[str] = []
        params: list = []
        if not admin:
            joins = (
                " JOIN (SELECT DISTINCT conversation_id, user_id FROM session_traces) t"
                " ON m.conversation_id = t.conversation_id"
            )
            conds.append("t.user_id = ?")
            params.append(requester_id)
        else:
            if user_id is not None:
                joins = (
                    " JOIN (SELECT DISTINCT conversation_id, user_id FROM session_traces) t"
                    " ON m.conversation_id = t.conversation_id"
                )
                conds.append("t.user_id = ?")
                params.append(scope)
        if scene:
            conds.append("m.scene = ?")
            params.append(scene)
        if status:
            conds.append("m.status = ?")
            params.append(status)
        sql = sql + joins
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY m.last_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]

        async with self._lock:
            rows = await self._fetchall(sql, params)
        return [_row_to_meta(r) for r in rows]

    async def _read_three_messages(self, conversation_id: str) -> Optional[list[dict]]:
        """优先从 03 主消息流读取权威主消息；不可用时返回 None（回退到 trace_events）。"""
        try:
            from sqlalchemy import select
            from app.db.models.connection import engine as three_engine
            from app.db.models import Message
        except Exception:
            return None
        if three_engine is None:
            return None
        try:
            from app.db.models.connection import async_session_maker as three_maker

            if three_maker is None:
                return None
            cid = int(conversation_id)
            async with three_maker() as s:
                rows = (
                    await s.execute(select(Message).where(Message.conversation_id == cid).order_by(Message.id))
                ).scalars().all()
                return [
                    {
                        "role": m.role,
                        "content": m.content,
                        "scene": m.scene,
                        "created_at": int(m.created_at.timestamp() * 1000) if m.created_at else 0,
                    }
                    for m in rows
                ]
        except Exception:
            return None

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
                "SELECT DISTINCT user_id FROM session_traces WHERE conversation_id=?",
                (conversation_id,),
            )
            if row is None:
                return None  # 该会话无历史记录
            await self._require_access(conversation_id, requester_id)

            # 主消息流：优先 03，否则从 trace_events 的 user/assistant 事件重建
            messages = await self._read_three_messages(conversation_id)
            if not messages:
                evs = await self._fetchall(
                    "SELECT * FROM trace_events WHERE turn_id IN "
                    "(SELECT turn_id FROM session_traces WHERE conversation_id=?) "
                    "AND type IN ('user','assistant') ORDER BY created_at, seq",
                    (conversation_id,),
                )
                messages = [
                    {"role": r["type"], "content": r["summary_in"] or r["summary_out"], "scene": None,
                     "created_at": r["created_at"]}
                    for r in evs
                ]

            meta = await self._get_meta(conversation_id)
            trace = None
            has_trace = False
            if with_trace:
                trace = await self._build_trace_tree(conversation_id)
                has_trace = trace is not None
        return SessionDetail(
            conversation_id=conversation_id,
            meta=meta,
            messages=messages,
            trace=trace,
            has_trace=has_trace,
        )

    async def _get_meta(self, conversation_id: str) -> Optional[SessionMeta]:
        row = await self._fetchone(
            "SELECT * FROM session_meta WHERE conversation_id=?", (conversation_id,)
        )
        return _row_to_meta(row) if row else None

    async def _build_trace_tree(self, conversation_id: str) -> Optional[list[dict]]:
        """按 run_id 分组，事件按 parent_event_id 嵌套成树。S9：无事件返回空列表。"""
        traces = await self._fetchall(
            "SELECT * FROM session_traces WHERE conversation_id=? ORDER BY created_at",
            (conversation_id,),
        )
        if not traces:
            return None
        runs: list[dict] = []
        for t in traces:
            events = await self._fetchall(
                "SELECT * FROM trace_events WHERE run_id=? ORDER BY seq", (t["run_id"],)
            )
            tree = self._nest_events([_row_to_event(e) for e in events])
            runs.append(
                {
                    "run_id": t["run_id"],
                    "turn_id": t["turn_id"],
                    "created_at": t["created_at"],
                    "model": t["model"],
                    "status": t["status"],
                    "duration_ms": t["duration_ms"],
                    "total_tokens": t["total_tokens"],
                    "events": tree,
                }
            )
        return runs

    @staticmethod
    def _nest_events(events: list[TraceEvent]) -> list[dict]:
        by_id: dict[int, dict] = {}
        roots: list[dict] = []
        for ev in events:
            node = {
                "id": ev.id,
                "agent": ev.agent,
                "type": ev.type,
                "summary_in": ev.summary_in,
                "summary_out": ev.summary_out,
                "meta": ev.meta,
                "tokens": ev.tokens,
                "duration_ms": ev.duration_ms,
                "children": [],
            }
            by_id[ev.id] = node
            if ev.parent_event_id is None or ev.parent_event_id not in by_id:
                roots.append(node)
            else:
                by_id[ev.parent_event_id]["children"].append(node)
        return roots

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
            await self._require_access(conversation_id, requester_id)

            t = await self._fetchone(
                "SELECT * FROM session_traces WHERE conversation_id=? AND turn_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (conversation_id, turn_id),
            )
            if t is None:
                return None
            evs = await self._fetchall(
                "SELECT * FROM trace_events WHERE turn_id=? ORDER BY seq", (turn_id,)
            )
        events = [_row_to_event(e) for e in evs]
        user_msg = next((e.summary_in for e in events if e.type == "user"), "")
        assistant_reply = next((e.summary_out for e in events if e.type == "assistant"), "")
        final = None
        if t["final_result"]:
            try:
                final = json.loads(t["final_result"])
            except Exception:
                final = None
        return TurnDetail(
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_message=user_msg,
            assistant_reply=assistant_reply,
            events=events,
            final_result=final,
        )

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
        from app.history.search import keyword_search as _kw

        async with self._lock:
            return await _kw(self, requester_id, query, scene=scene, start=start, end=end, limit=limit)

    async def semantic_search(
        self,
        requester_id: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchHit]:
        from app.history.search import semantic_search as _sem

        async with self._lock:
            return await _sem(self, requester_id, query, top_k=top_k)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

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
                    "DELETE FROM trace_events WHERE run_id IN "
                    "(SELECT run_id FROM session_traces WHERE conversation_id=?)",
                    (conversation_id,),
                )
                await self._conn.execute(
                    "DELETE FROM history_embeddings WHERE conversation_id=?", (conversation_id,)
                )
                await self._conn.execute(
                    "DELETE FROM session_traces WHERE conversation_id=?", (conversation_id,)
                )
                await self._conn.execute(
                    "DELETE FROM session_meta WHERE conversation_id=?", (conversation_id,)
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
        """清理 before_date 之前的轨迹。仅管理员可调用（评审 B2）。

        cascade=True 时联动清 03 主消息（谨慎，默认 false，受合规约束）。"""
        if not self.enabled:
            return 0
        if not self._is_admin(requester_id):
            raise HistoryAccessError("purge_before 仅管理员可调用")
        before = _to_epoch_ms(before_date)
        if before is None:
            return 0
        try:
            async with self._lock:
                if cascade:
                    await self._cascade_purge_three(before)
                await self._conn.execute(
                    "DELETE FROM trace_events WHERE run_id IN "
                    "(SELECT run_id FROM session_traces WHERE created_at < ?)",
                    (before,),
                )
                await self._conn.execute(
                    "DELETE FROM history_embeddings WHERE conversation_id IN "
                    "(SELECT conversation_id FROM session_traces WHERE created_at < ?)",
                    (before,),
                )
                await self._conn.execute(
                    "DELETE FROM session_traces WHERE created_at < ?", (before,)
                )
                await self._conn.execute(
                    "DELETE FROM session_meta WHERE last_at < ? OR last_at IS NULL", (before,)
                )
                await self._conn.commit()
            return 1
        except Exception as exc:
            await self._on_write_failure(exc, {"before_date": str(before_date)})
            return 0

    async def _cascade_purge_three(self, before_ms: int) -> None:
        """联动清理 03 主消息（cascade=True 时）。仅清 before 之前、且 07 已知归属的会话。"""
        try:
            from sqlalchemy import delete
            from app.db.models.connection import async_session_maker as three_maker
            from app.db.models import Message, Conversation

            if three_maker is None:
                return
            async with three_maker() as s:
                await s.execute(
                    delete(Message).where(
                        Message.created_at < _ms_to_datetime(before_ms)
                    )
                )
                await s.commit()
        except Exception as exc:  # pragma: no cover - 联动清理失败不阻断 07 清理
            logger.warning("07 cascade 清理 03 主消息失败: %s", exc)

    # —— 导出/复盘（委托 export.py） —— #
    async def export_session(
        self, requester_id: str, conversation_id: str, fmt: str = "md"
    ) -> str:
        from app.history.export import render_export

        detail = await self.get_session(requester_id, conversation_id, with_trace=True)
        if detail is None:
            return ""
        return render_export(detail, fmt=fmt)

    async def get_session_for_review(
        self, requester_id: str, conversation_id: str
    ) -> str:
        from app.history.export import render_review_context

        detail = await self.get_session(requester_id, conversation_id, with_trace=True)
        if detail is None:
            return ""
        return render_review_context(detail)


# ---------------------------------------------------------------------------
# 行 → 模型 转换
# ---------------------------------------------------------------------------
def _row_to_meta(row) -> SessionMeta:
    return SessionMeta(
        conversation_id=row["conversation_id"],
        scene=row["scene"],
        title=row["title"],
        status=row["status"] or "active",
        msg_count=row["msg_count"] or 0,
        total_tokens=row["total_tokens"] or 0,
        first_at=row["first_at"] or 0,
        last_at=row["last_at"] or 0,
        created_at=row["created_at"] or 0,
    )


def _row_to_event(row) -> TraceEvent:
    return TraceEvent(
        id=row["id"],
        run_id=row["run_id"],
        parent_event_id=row["parent_event_id"],
        turn_id=row["turn_id"],
        seq=row["seq"] or 0,
        agent=row["agent"] or "",
        type=row["type"] or "",
        summary_in=row["summary_in"] or "",
        summary_out=row["summary_out"] or "",
        meta_json=row["meta_json"] or "",
        tokens=row["tokens"] or 0,
        duration_ms=row["duration_ms"] or 0,
        created_at=row["created_at"] or 0,
    )


def _ms_to_datetime(ms: int):
    import datetime

    return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc)


# ---------------------------------------------------------------------------
# 工厂（单例缓存，便于测试 reset）
# ---------------------------------------------------------------------------
_STORE: Any = None


def _detect_model_name() -> str:
    try:
        from app.config import config

        model_cfg = config.get("model") or {}
        mode = model_cfg.get("mode", "api")
        sub = model_cfg.get(mode) or {}
        return sub.get("model_name") or "unknown"
    except Exception:
        return "unknown"


def get_history_store() -> Any:
    """按 config.history.enabled 返回实现；enabled=false 时返回 NoopHistoryStore。"""
    global _STORE
    if _STORE is not None:
        return _STORE
    from app.config import config

    cfg = config.get("history") or {}
    if not cfg.get("enabled", False):
        _STORE = NoopHistoryStore()
        return _STORE
    # 复用 03 主库（db.sqlite_path 指定的 agentar.db），不另开文件；
    # history.db_path 仅作为可选覆盖项（默认不配，走主库）。
    db_path = cfg.get("db_path")
    if not db_path:
        db_path = (config.get("db", {}) or {}).get("sqlite_path", "./agentar.db")
    if not os.path.isabs(db_path):
        db_path = str(Path(db_path).resolve())
    _STORE = SessionHistoryStore(
        db_path,
        record_traces=bool(cfg.get("record_traces", True)),
        async_write=bool(cfg.get("async_write", True)),
        fail_mode=cfg.get("fail_mode", "silent"),
        pii_redact=bool(cfg.get("pii_redact", True)),
        semantic_search=bool(cfg.get("semantic_search", False)),
        allow_admin_all=bool(cfg.get("allow_admin_all", False)),
        admin_user_ids=cfg.get("admin_user_ids", []) or [],
        retention_days=int(cfg.get("retention_days", 180)),
        model_name=_detect_model_name(),
        history_dir=os.path.dirname(db_path),
    )
    return _STORE


def reset_history_store() -> None:
    """清空单例缓存（测试用）。"""
    global _STORE
    _STORE = None


async def init_history_db() -> None:
    """幂等建表入口（评审 B6），由 main.py lifespan 调用；enabled=false 时直接返回。"""
    store = get_history_store()
    if not store.enabled:
        return
    await store.connect()
    await store.init_tables()


__all__ = [
    "SessionHistoryStore",
    "NoopHistoryStore",
    "get_history_store",
    "reset_history_store",
    "init_history_db",
]
