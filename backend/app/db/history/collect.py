"""07-事件构造器：从 AgentResult（+ 02 未来暴露的 step 事件流）构建 trace_events。

v1（02 未暴露 step 钩子）：从 ``AgentResult`` 即可拿到
  - user 主消息事件（来自 hook 传入的 user_message）
  - assistant 主消息事件（result.reply）
  - review 事件（result.compliance_notes，建议式审核结论）
  - risk 事件（result.risk_flags，建议式风控结论）

v2（02 扩展 step 钩子后）：通过 ``extra_events`` 注入 Coordinator→RAG→专家→审核→风控的
细粒度 step 事件（tool_call / rag_hit / 各 agent 步骤 / error），达成完整回放（评审 B1）。

所有事件在写入前由 store 统一脱敏（评审 S2）。
"""
from __future__ import annotations

import time

from app.db.history.models import TraceEvent


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_events(
    result: object,
    *,
    user_message: str = "",
    extra_events: list[TraceEvent] | None = None,
) -> list[TraceEvent]:
    """根据 AgentResult（v1）与可选 extra_events（v2）构造 trace_events 列表。"""
    events: list[TraceEvent] = []
    ts = _now_ms()

    if user_message:
        events.append(
            TraceEvent(agent="user", type="user", summary_in=user_message, created_at=ts)
        )

    reply = getattr(result, "reply", "") or ""
    if reply:
        events.append(
            TraceEvent(agent="assistant", type="assistant", summary_out=reply, created_at=ts)
        )

    # v1 即可记录的审核 / 风控事件（来自 AgentResult，不依赖 02 step 钩子）
    for note in getattr(result, "compliance_notes", None) or []:
        if note:
            events.append(
                TraceEvent(agent="review", type="review", summary_out=str(note), created_at=ts)
            )
    for flag in getattr(result, "risk_flags", None) or []:
        if flag:
            events.append(
                TraceEvent(agent="risk", type="risk", summary_out=str(flag), created_at=ts)
            )

    # v2：02 暴露 step 钩子后注入的细粒度事件（保持顺序，seq 由 store 写入时分配）
    for ev in extra_events or []:
        events.append(ev)

    return events
