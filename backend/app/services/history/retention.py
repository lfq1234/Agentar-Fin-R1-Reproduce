"""07-留存与清理：把 retention_days 策略落到 purge_before。

清理为低频运维操作，建议由脚本/定时任务触发，不在请求链路同步执行（需求 FR7）。
`delete_session` / `purge_before` 的实际 DB 删除在 store 内实现；本模块提供策略入口。
"""
from __future__ import annotations

import time

from app.services.history.models import HistoryStore


async def apply_retention(
    store: HistoryStore,
    *,
    requester_id: str,
    retention_days: int,
    cascade: bool = False,
) -> int:
    """按 retention_days 清理更早的轨迹；仅管理员可调用（委托 store.purge_before）。

    返回被清理的会话批次数（0 或 1；store.purge_before 当前按时间一次性清理）。
    """
    before_ms = int(time.time() * 1000) - retention_days * 86400 * 1000
    return await store.purge_before(requester_id, before_ms, cascade=cascade)
