"""07-采集钩子集成测试：无侵入包裹 chat_service.chat。

验证：
- install_history_tracing 能在不修改 03 源码的情况下记录历史；
- 主链路返回原 resp 且不被采集异常阻断（评审：无侵入 + 降级不阻塞）；
- resp.conversation_id 为 None / 无 user_id 时安全跳过；
- scene 取自请求侧（ChatResponse 无 scene 字段，回归校验 resp.scene 已修复）。

说明：原实现向 sys.modules 注入假 app.services.chat_service 以避免加载 agentscope，
但该方式在 pytest 收集期会被其它模块（如 test_main 导入 app.main 时把 chat_service
绑定到 app.services 包命名空间）污染，导致全量跑时 hooks 拿到的是真实模块。
本版改为直接 monkeypatch 真实 chat_service.chat 为桩函数——更稳妥、且不污染
其它测试模块的顶层导入（agentscope 在 envs/default 已安装，且 hooks 对其为惰性导入）。
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest
import pytest_asyncio

from app.db.history import hooks
from app.db.history import store as storemod
from app.services import chat_service as chat_service_mod


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
    # 让 record_run 的兜底 INSERT 能满足 conversations.user_id 外键
    for uid in (1, 2):
        await s._conn.execute(
            "INSERT OR IGNORE INTO users (id, username, created_at, updated_at) "
            "VALUES (?, ?, '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00')",
            (uid, f"u{uid}"),
        )
    await s._conn.commit()
    storemod._store = s  # 让 hooks 内的 get_history_store() 拿到本测试实例
    yield s
    storemod._store = None
    await s.close()


async def _original_chat(req, db):
    return _Resp(
        reply="这是助手的回答",
        conversation_id=42,
        compliance_notes=["合规结论：通过"],
        risk_flags=["低风险"],
    )


@pytest.mark.asyncio
async def test_traced_chat_records_and_returns_resp(store, monkeypatch):
    monkeypatch.setattr(chat_service_mod, "chat", _original_chat)
    traced = hooks.install_history_tracing()

    req = _Req(user_id=1, message="请解释GDP", scene="qa")
    resp = await traced(req, None)

    # 主链路返回原 resp，不被阻断
    assert resp.reply == "这是助手的回答"
    # 历史已记录：兜底路径会补一条 assistant 消息并把 trace 挂上
    detail = await store.get_session(1, "42")
    assert detail is not None
    roles = {m["role"] for m in detail.messages}
    assert "assistant" in roles
    assert any(e["type"] == "review" for e in detail.trace[0]["events"])


@pytest.mark.asyncio
async def test_traced_chat_scene_from_request(store, monkeypatch):
    # 回归：scene 必须来自 req（ChatResponse 无 scene 字段），且不能抛 AttributeError
    monkeypatch.setattr(chat_service_mod, "chat", _original_chat)
    traced = hooks.install_history_tracing()
    req = _Req(user_id=1, message="hi", scene="invest")
    resp = await traced(req, None)
    assert resp.reply == "这是助手的回答"
    detail = await store.get_session(1, "42")
    assert detail is not None
    assert detail.meta.scene == "invest"


@pytest.mark.asyncio
async def test_traced_chat_skips_when_no_conversation(store, monkeypatch):
    async def _no_conv(req, db):
        return _Resp(reply="x", conversation_id=None)

    monkeypatch.setattr(chat_service_mod, "chat", _no_conv)
    traced = hooks.install_history_tracing()
    req = _Req(user_id=1, message="hi", scene="qa")
    resp = await traced(req, None)
    assert resp.conversation_id is None
    # 不应有任何记录
    assert await store.get_session(1, "None") is None


@pytest.mark.asyncio
async def test_traced_chat_async_write_schedules_task(store, monkeypatch):
    # 异步写入路径：create_task 后任务应在事件循环中完成，且主链路不阻塞
    monkeypatch.setattr(chat_service_mod, "chat", _original_chat)
    # 重建为 async_write=True 的 store
    await store.close()
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "history.db")
    s2 = storemod.SessionHistoryStore(
        db, record_traces=True, fail_mode="silent", async_write=True
    )
    await s2.connect()
    await s2.init_tables()
    await s2._conn.execute(
        "INSERT OR IGNORE INTO users (id, username, created_at, updated_at) "
        "VALUES (?, ?, '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00')",
        (2, "u2"),
    )
    await s2._conn.commit()
    storemod._store = s2
    traced = hooks.install_history_tracing()
    req = _Req(user_id=2, message="异步写入测试", scene="qa")
    resp = await traced(req, None)
    # 让后台写任务跑完
    await asyncio.sleep(0.05)
    detail = await s2.get_session(2, "42")
    assert detail is not None
    # 兜底写入的是助手消息（含 trace），其内容即 _original_chat 的 reply
    assert "这是助手的回答" in detail.messages[0]["content"]
    await s2.close()  # 关闭本测试自建的 store，避免连接泄漏导致的事件循环关闭告警
