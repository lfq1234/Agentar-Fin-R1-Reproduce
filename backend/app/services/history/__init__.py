"""07-会话历史记录（Session History & Trace）包入口。

导出工厂与核心类型；提供 ``init_history_db``（建表入口）与 ``install_history_tracing``
（无侵入接线 03 chat_service）。配置段见 config.yaml 的 ``history:``。
"""
from __future__ import annotations

from app.services.history.models import (
    HistoryAccessError,
    HistoryStore,
    HistoryEmbedding,
    SearchHit,
    SessionDetail,
    SessionMeta,
    SessionTrace,
    TraceEvent,
    TurnDetail,
)
from app.services.history.store import (
    NoopHistoryStore,
    SessionHistoryStore,
    get_history_store,
    init_history_db,
    reset_history_store,
)
from app.services.history.hooks import install_history_tracing

__all__ = [
    "HistoryStore",
    "HistoryAccessError",
    "TraceEvent",
    "SessionTrace",
    "SessionMeta",
    "HistoryEmbedding",
    "SearchHit",
    "SessionDetail",
    "TurnDetail",
    "SessionHistoryStore",
    "NoopHistoryStore",
    "get_history_store",
    "init_history_db",
    "reset_history_store",
    "install_history_tracing",
]
