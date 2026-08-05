"""07-会话历史记录：数据模型与契约。

设计取舍（评审 B4 / S6）：
- 07 使用**独立 SQLite**（history.db），与 03 库解耦；因此 07 的模型用普通 dataclass 定义，
  不注入 03 的全局 SQLModel.metadata，避免跨模块元数据污染。
- 时间字段统一为 ``INTEGER``（epoch 毫秒，UTC），避免 VARCHAR ISO 字符串比较在跨时区/本地时区
  混用时出现漏删/错排（评审 B4）。
- ``turn_id`` 由 07 在每次 ``chat()`` 调用时生成（一个 user→assistant 交互 = 一个 turn），落到
  ``session_traces.turn_id`` 与 ``trace_events.turn_id``，作为回放单轮的稳定键（评审 B5）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


class HistoryAccessError(Exception):
    """权限/归属校验失败：非本人且非管理员尝试访问他人会话。"""


# ---------------------------------------------------------------------------
# 持久化数据模型（dataclass，对应 schema.sql）
# ---------------------------------------------------------------------------
@dataclass
class TraceEvent:
    """轨迹事件树中的一个节点（agent 步骤 / tool / RAG 命中 / 审核 / 风控 / 错误 / 主消息）。"""

    id: Optional[int] = None
    run_id: str = ""
    parent_event_id: Optional[int] = None
    turn_id: str = ""
    seq: int = 0
    agent: str = ""  # user | assistant | Coordinator | rag | 专家 | review | risk | ...
    type: str = ""   # user|assistant|tool_call|tool_result|rag_hit|review|risk|error
    summary_in: str = ""
    summary_out: str = ""
    meta_json: str = ""  # 结构化扩展：tool 名/参数、RAG 命中溯源、错误类型等
    tokens: int = 0
    duration_ms: int = 0
    created_at: int = 0  # epoch 毫秒

    @property
    def meta(self) -> dict:
        if not self.meta_json:
            return {}
        try:
            return json.loads(self.meta_json)
        except Exception:
            return {}

    @meta.setter
    def meta(self, value: dict) -> None:
        self.meta_json = json.dumps(value, ensure_ascii=False) if value else ""


@dataclass
class SessionTrace:
    """一次 ``run()`` 的执行轨迹头。"""

    run_id: str
    turn_id: str = ""
    conversation_id: str = ""
    user_id: str = ""
    scene: Optional[str] = None
    created_at: int = 0  # epoch 毫秒
    duration_ms: int = 0
    total_tokens: int = 0
    model: str = ""
    status: str = "ok"  # ok | error
    final_result: str = ""  # JSON 序列化 AgentResult


@dataclass
class SessionMeta:
    """会话级扩展元信息（不 ALTER 03 conversations，独立表）。"""

    conversation_id: str
    scene: Optional[str] = None
    title: Optional[str] = None
    status: str = "active"  # active | archived | deleted
    msg_count: int = 0
    total_tokens: int = 0
    first_at: int = 0  # epoch 毫秒
    last_at: int = 0  # epoch 毫秒
    created_at: int = 0


@dataclass
class HistoryEmbedding:
    """可选语义检索：历史消息向量（开启 history.semantic_search 时启用）。"""

    message_id: str
    conversation_id: str = ""
    embedding: str = ""  # JSON 数组
    dim: int = 0
    model: str = ""  # embedder 版本（切换 embedder 需重建，评审 S5）


# ---------------------------------------------------------------------------
# 查询/回放返回结构
# ---------------------------------------------------------------------------
@dataclass
class SearchHit:
    conversation_id: str
    turn_id: str
    event_id: Optional[int]
    snippet: str
    score: float
    type: str
    created_at: int


@dataclass
class TurnDetail:
    """单轮（一次 user→assistant）完整内容。"""

    conversation_id: str
    turn_id: str
    user_message: str = ""
    assistant_reply: str = ""
    events: list[TraceEvent] = field(default_factory=list)
    final_result: Optional[dict] = None


@dataclass
class SessionDetail:
    """一次会话的完整回放。"""

    conversation_id: str
    meta: Optional[SessionMeta] = None
    messages: list[dict] = field(default_factory=list)  # 主消息流 [{role,content,scene,...}]
    trace: Any = None  # 按 run_id 分组的事件树（可序列化结构）
    has_trace: bool = False


# ---------------------------------------------------------------------------
# 存储契约（Protocol）：SQLite 实现 / Noop 实现共用
# ---------------------------------------------------------------------------
@runtime_checkable
class HistoryStore(Protocol):
    """07 历史存储统一契约。"""

    enabled: bool

    async def connect(self) -> None: ...

    async def init_tables(self) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    async def list_sessions(
        self,
        requester_id: str,
        *,
        user_id: Optional[str] = None,
        scene: Optional[str] = None,
        status: str = "active",
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionMeta]: ...

    async def get_session(
        self,
        requester_id: str,
        conversation_id: str,
        *,
        with_trace: bool = True,
    ) -> Optional[SessionDetail]: ...

    async def get_turn(
        self,
        requester_id: str,
        conversation_id: str,
        turn_id: str,
    ) -> Optional[TurnDetail]: ...

    async def keyword_search(
        self,
        requester_id: str,
        query: str,
        *,
        scene: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 20,
    ) -> list[SearchHit]: ...

    async def semantic_search(
        self,
        requester_id: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SearchHit]: ...

    async def export_session(
        self,
        requester_id: str,
        conversation_id: str,
        fmt: str = "md",
    ) -> str: ...

    async def get_session_for_review(
        self,
        requester_id: str,
        conversation_id: str,
    ) -> str: ...

    async def delete_session(
        self,
        requester_id: str,
        conversation_id: str,
    ) -> int: ...

    async def purge_before(
        self,
        requester_id: str,
        before_date: Any,
        *,
        cascade: bool = False,
    ) -> int: ...
