"""07-采集钩子：无侵入包裹 03 ``chat_service.chat``，旁路异步写入历史，失败不阻塞主链路。

评审约束落点：
- 无侵入：install_history_tracing() 在运行时替换 ``chat_service.chat`` 为包裹版本，
  不修改 03 源码（技术文档 §4.2）。
- 旁路异步（async_write=true 时 create_task，不阻塞返回 reply）+ 降级（写入失败由 store
  兜底 silent/local，主链路照常返回）。
- 采集逻辑整体包在 try/except 内：任何异常（含字段缺失）都吞掉，绝不阻断主链路返回 reply。
- v1（02 未暴露 step 钩子）：仅记录「头 + user/assistant + review/risk 事件」；02 扩展后
  通过 collect.build_events(extra_events=...) 补全细粒度事件（评审 B1）。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from app.db.history.collect import build_events
from app.db.history.models import TraceEvent
from app.db.history.store import get_history_store


async def _swallow(coro: Any) -> None:
    """吞掉异步写任务的异常，避免 create_task 未被 await 时静默抛到事件循环。"""
    try:
        await coro
    except Exception:
        pass


def install_history_tracing() -> Any:
    """包裹 ``app.services.chat_service.chat`` 为带历史采集的版本。

    返回包裹后的函数（便于测试断言）；重复调用会重复包裹（幂等于行为，但不叠加）。
    """
    from app.services import chat_service

    original = chat_service.chat

    async def traced_chat(req: Any, db: Any, **kwargs: Any):
        store = get_history_store()
        t0 = time.monotonic()
        # 透传额外关键字（如 user_id），使路由显式传入的归属信息直达原 chat
        resp = await original(req, db, **kwargs)
        # 采集逻辑整体包在 try 内：任何异常（含字段缺失）都不阻断主链路
        try:
            # 延迟导入：避免在 import app.db.history 时拉起 agentscope 重依赖
            from app.agent.system import AgentResult

            if not store.enabled:
                return resp
            if resp.conversation_id is None or req.user_id is None:
                return resp

            run_id = uuid.uuid4().hex
            turn_id = uuid.uuid4().hex
            duration_ms = int((time.monotonic() - t0) * 1000)
            user_id = str(req.user_id)

            # 注意：ChatResponse 无 scene 字段，scene 取自请求侧（会话 scene 继承请求）。
            result = AgentResult(
                reply=resp.reply,
                compliance_notes=resp.compliance_notes,
                risk_flags=resp.risk_flags,
                agent_trace=getattr(resp, "agent_trace", None),
            )
            # 02 扩展（评审 B1 / v2）：把多智能体细粒度步骤转为 trace_events 落库，
            # 专家对话（route/rag/expert_opinion/synthesize/revise）一并存入本次会话轨迹。
            # 用 getattr 容忍未携带该字段的响应（如测试桩 / 旧调用方），缺失时退化为 v1 记录。
            extra_events = [
                TraceEvent(
                    agent=step.get("agent", ""),
                    type=step.get("type", ""),
                    summary_out=step.get("content", ""),
                    meta_json=json.dumps(step.get("meta") or {}, ensure_ascii=False),
                )
                for step in (getattr(resp, "agent_trace", None) or [])
            ]
            events = build_events(result, user_message=req.message, extra_events=extra_events)
            coro = store.record_run(
                conversation_id=str(resp.conversation_id),
                user_id=user_id,
                scene=req.scene,
                run_id=run_id,
                turn_id=turn_id,
                duration_ms=duration_ms,
                model=store.model_name,
                result=result,
                events=events,
                user_message=req.message,
            )
            if store.async_write:
                asyncio.create_task(_swallow(coro))
            else:
                try:
                    await coro
                except Exception:
                    pass
        except Exception:
            pass
        return resp

    chat_service.chat = traced_chat
    return traced_chat
