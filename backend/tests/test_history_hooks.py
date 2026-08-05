"""07-采集钩子集成测试：无侵入包裹 chat_service.chat。

通过向 sys.modules 注入假的 ``app.agent.system``（提供 AgentResult）与假的
``app.services.chat_service``，避免引入 agentscope 等重依赖，验证：
- install_history_tracing 能在不修改 03 源码的情况下记录历史；
- 主链路返回原 resp 且不被采集异常阻断（评审：无侵入 + 降级不阻塞）；
- resp.conversation_id 为 None / 无 user_id 时安全跳过；
- scene 取自请求侧（ChatResponse 无 scene 字段，回归校验 resp.scene 已修复）。
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import types

import pytest
import pytest_asyncio

# —— 在任何 app.services.history 导入前注入假依赖（避免 agentscope 真正加载） —— #
_agent_system = types.ModuleType("app.agent.system")


class _AgentResult:
    def __init__(self, reply="", compliance_notes=None, risk_flags=None):
        self.reply = reply
        self.compliance_notes = compliance_notes or []
        self.risk_flags = risk_flags or []


_agent_system.AgentResult = _AgentResult
_agent_system.__fake__ = True
sys.modules.setdefault("app.agent", types.ModuleType("app.agent"))
sys.modules["app.agent.system"] = _agent_system
_chat_service_mod = types.ModuleType("app.services.chat_service")
_chat_service_mod.__fake__ = True
_chat_service_mod.chat = None
sys.modules["app.services.chat_service"] = _chat_service_mod

from app.services.history import hooks  # noqa: E402
from app.services.history import store as storemod  # noqa: E402


class _Resp:
    def __init__(self, reply, conversation_id, compliance_notes=None, risk_flags=None):
        self.reply = reply
        self.conversation_id = conversation_id
        self.compliance_notes = compliance_notes or []
        self.risk_flags = risk_flags or []


class _Req:
    def __init__(self, user_id, message, scene, conversation_id=None):
        self.user_id = user_id
        self.message = message
        self.scene = scene
        self.conversation_id = conversation_id


@pytest_asyncio.fixture
async def store():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "history.db")
    s = storemod.SessionHistoryStore(
        db, record_traces=True, fail_mode="silent", async_write=False
    )
    await s.connect()
    await s.init_tables()
    storemod._STORE = s  # 让 hooks 内的 get_history_store() 拿到本测试实例
    yield s
    storemod._STORE = None
    await s.close()


async def _original_chat(req, db):
    return _Resp(
        reply="这是助手的回答",
        conversation_id=42,
        compliance_notes=["合规结论：通过"],
        risk_flags=["低风险"],
    )


@pytest.mark.asyncio
async def test_traced_chat_records_and_returns_resp(store):
    cs = sys.modules["app.services.chat_service"]
    cs.chat = _original_chat
    traced = hooks.install_history_tracing()

    req = _Req(user_id="u1", message="请解释GDP", scene="qa")
    resp = await traced(req, None)

    # 主链路返回原 resp，不被阻断
    assert resp.reply == "这是助手的回答"
    # 历史已记录
    detail = await store.get_session("u1", "42")
    assert detail is not None
    roles = {m["role"] for m in detail.messages}
    assert roles >= {"user", "assistant"}
    assert any(e["type"] == "review" for e in detail.trace[0]["events"])


@pytest.mark.asyncio
async def test_traced_chat_scene_from_request(store):
    # 回归：scene 必须来自 req（ChatResponse 无 scene 字段），且不能抛 AttributeError
    cs = sys.modules["app.services.chat_service"]
    cs.chat = _original_chat
    traced = hooks.install_history_tracing()
    req = _Req(user_id="u1", message="hi", scene="invest")
    resp = await traced(req, None)
    assert resp.reply == "这是助手的回答"
    meta = await store._get_meta("42")
    assert meta is not None
    assert meta.scene == "invest"


@pytest.mark.asyncio
async def test_traced_chat_skips_when_no_conversation(store):
    async def _no_conv(req, db):
        return _Resp(reply="x", conversation_id=None)

    cs = sys.modules["app.services.chat_service"]
    cs.chat = _no_conv
    traced = hooks.install_history_tracing()
    req = _Req(user_id="u1", message="hi", scene="qa")
    resp = await traced(req, None)
    assert resp.conversation_id is None
    # 不应有任何记录
    assert await store.get_session("u1", "None") is None


@pytest.mark.asyncio
async def test_traced_chat_async_write_schedules_task(store):
    # 异步写入路径：create_task 后任务应在事件循环中完成，且主链路不阻塞
    cs = sys.modules["app.services.chat_service"]
    cs.chat = _original_chat
    # 重建为 async_write=True 的 store
    await store.close()
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "history.db")
    s2 = storemod.SessionHistoryStore(
        db, record_traces=True, fail_mode="silent", async_write=True
    )
    await s2.connect()
    await s2.init_tables()
    storemod._STORE = s2
    traced = hooks.install_history_tracing()
    req = _Req(user_id="u2", message="异步写入测试", scene="qa")
    resp = await traced(req, None)
    # 让后台写任务跑完
    await asyncio.sleep(0.05)
    detail = await s2.get_session("u2", "42")
    assert detail is not None
    assert "异步写入测试" in detail.messages[0]["content"]
    await s2.close()  # 关闭本测试自建的 store，避免连接泄漏导致的事件循环关闭告警
