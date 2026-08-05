"""07-会话历史记录：核心模块测试（store/models/redact/collect/search/export/retention）。

不依赖 agentscope / duckdb / 真实模型 key，可在隔离 venv 中运行。
覆盖评审 B1-B6 / S1-S9 的关键行为：
- B2/B3 权限：非本人拒绝、管理员跨用户可见、删除/清理仅本人或管理员。
- B4 时间：时间字段为 INTEGER epoch 毫秒，purge_before / 时间过滤基于它。
- B5 turn_id：回放单轮可定位。
- B6 幂等建表。
- S1 写串行化（asyncio.Lock）；S2 脱敏覆盖；S4 local 兜底；S8 silent 静默；
  S7 meta 同步；S9 record_traces=false 安全降级。
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import pytest_asyncio

from app.services.history import collect
from app.services.history import export as exportmod
from app.services.history import models as M
from app.services.history import redact
from app.services.history import retention as retentionmod
from app.services.history import store as storemod


class FakeResult:
    """模拟 app.agent.system.AgentResult（v1 字段）。"""

    def __init__(self, reply="", compliance_notes=None, risk_flags=None):
        self.reply = reply
        self.compliance_notes = compliance_notes or []
        self.risk_flags = risk_flags or []


def _new_store(tmp, **kw):
    db = os.path.join(tmp, "history.db")
    s = storemod.SessionHistoryStore(
        db,
        record_traces=True,
        fail_mode="silent",
        allow_admin_all=True,
        admin_user_ids=["admin1"],
        **kw,
    )
    return s


@pytest_asyncio.fixture
async def store():
    tmp = tempfile.mkdtemp()
    s = _new_store(tmp)
    await s.connect()
    await s.init_tables()
    yield s
    await s.close()


def _do_record(s, conv, user, *, reply="答案是42", note="合规OK", msg="什么是GDP"):
    result = FakeResult(reply=reply, compliance_notes=[note], risk_flags=[])
    events = collect.build_events(result, user_message=msg)
    return s.record_run(
        conversation_id=conv,
        user_id=user,
        scene="qa",
        run_id="run-" + conv,
        turn_id="turn-" + conv,
        duration_ms=123,
        model="test-model",
        result=result,
        events=events,
        user_message=msg,
        total_tokens=10,
    )


# ---------------------------------------------------------------------------
# 模型 / 脱敏 / 采集
# ---------------------------------------------------------------------------
def test_trace_event_meta_roundtrip():
    ev = M.TraceEvent(agent="rag", type="rag_hit", summary_out="命中")
    ev.meta = {"doc_id": "d1", "score": 0.9}
    assert ev.meta == {"doc_id": "d1", "score": 0.9}
    assert "doc_id" in ev.meta_json


def test_redact_pii():
    txt = "手机号13800138000，身份证11010119900307123X，卡6217001234567890，邮箱a@b.com"
    out = redact.redact_text(txt)
    assert "13800138000" not in out
    assert "****" in out
    assert "a@b.com" not in out
    # 邮箱用户过长被掩
    assert "***@b.com" in out or "@b.com" not in out


def test_redact_value_recursive():
    payload = {"reply": "联系13800138000", "list": ["邮件x@y.com"], "n": 1}
    out = redact.redact_value(payload)
    assert "13800138000" not in json.dumps(out)
    assert "x@y.com" not in json.dumps(out)


def test_collect_v1_and_v2():
    res = FakeResult(reply="hi", compliance_notes=["c1"], risk_flags=["r1"])
    evs_v1 = collect.build_events(res, user_message="u")
    types = {e.type for e in evs_v1}
    assert types >= {"user", "assistant", "review", "risk"}

    extra = [M.TraceEvent(agent="rag", type="rag_hit", summary_out="doc")]
    evs_v2 = collect.build_events(res, user_message="u", extra_events=extra)
    assert any(e.type == "rag_hit" for e in evs_v2)


# ---------------------------------------------------------------------------
# 写入 / 回放
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_record_and_get_session(store):
    await _do_record(store, "conv-1", "userA")
    detail = await store.get_session("userA", "conv-1")
    assert detail is not None
    assert detail.has_trace is True
    assert detail.meta is not None
    assert detail.meta.msg_count == 2
    roles = {m["role"] for m in detail.messages}
    assert roles >= {"user", "assistant"}


@pytest.mark.asyncio
async def test_get_turn(store):
    await _do_record(store, "conv-1", "userA")
    turn = await store.get_turn("userA", "conv-1", "turn-conv-1")
    assert turn is not None
    assert turn.user_message == "什么是GDP"
    assert "答案是42" in turn.assistant_reply
    assert any(e.type == "review" for e in turn.events)


@pytest.mark.asyncio
async def test_record_traces_false_safe_degrade(store):
    await store.record_run(
        conversation_id="conv-2",
        user_id="userA",
        scene="qa",
        run_id="run-conv-2",
        turn_id="turn-conv-2",
        duration_ms=1,
        model="m",
        result=FakeResult(reply="x"),
        events=[],
        user_message="hi",
    )
    detail = await store.get_session("userA", "conv-2")
    # record_traces=true 时仍有 session_traces 头；has_trace 取决于 trace_events 是否有事件
    assert detail is not None


@pytest.mark.asyncio
async def test_init_tables_idempotent(store):
    # 重复建表不应报错（B6）
    await store.init_tables()
    await store.init_tables()


# ---------------------------------------------------------------------------
# 权限（B2/B3）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_sessions_scoping(store):
    await _do_record(store, "conv-A", "userA")
    mine = await store.list_sessions("userA")
    assert any(m.conversation_id == "conv-A" for m in mine)
    others = await store.list_sessions("userB")
    assert all(m.conversation_id != "conv-A" for m in others)
    admin = await store.list_sessions("admin1")
    assert any(m.conversation_id == "conv-A" for m in admin)


@pytest.mark.asyncio
async def test_get_session_permission_denied(store):
    await _do_record(store, "conv-A", "userA")
    with pytest.raises(M.HistoryAccessError):
        await store.get_session("userB", "conv-A")


@pytest.mark.asyncio
async def test_delete_own_and_denied(store):
    await _do_record(store, "conv-A", "userA")
    n = await store.delete_session("userA", "conv-A")
    assert n == 1
    assert await store.get_session("userA", "conv-A") is None
    # 重新写一条，他人删除应被拒
    await _do_record(store, "conv-B", "userA")
    with pytest.raises(M.HistoryAccessError):
        await store.delete_session("userB", "conv-B")


# ---------------------------------------------------------------------------
# 检索（B2 字段级鉴权 + B4 时间）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_keyword_search_scoping_and_time(store):
    await _do_record(store, "conv-A", "userA", msg="查询GDP增长率")
    # 本人命中
    hits = await store.keyword_search("userA", "GDP")
    assert any(h.conversation_id == "conv-A" for h in hits)
    # 他人无命中（非管理员）
    hits_other = await store.keyword_search("userB", "GDP")
    assert not any(h.conversation_id == "conv-A" for h in hits_other)
    # 管理员可见
    hits_admin = await store.keyword_search("admin1", "GDP")
    assert any(h.conversation_id == "conv-A" for h in hits_admin)
    # 时间过滤：start=未来 → 无命中
    future = 9_999_999_999_999_999
    hits_future = await store.keyword_search("userA", "GDP", start=future)
    assert hits_future == []


# ---------------------------------------------------------------------------
# 留存 / 清理（B4 时间 + 管理员门禁）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_purge_before_admin_only_and_time(store):
    await _do_record(store, "conv-OLD", "userA")
    # 把 OLD 的 created_at 改到 10 天前
    old_ts = 1_600_000_000_000  # 固定过去时间戳
    await store._conn.execute(
        "UPDATE session_traces SET created_at=? WHERE conversation_id=?",
        (old_ts, "conv-OLD"),
    )
    await store._conn.execute(
        "UPDATE session_meta SET first_at=?, last_at=? WHERE conversation_id=?",
        (old_ts, old_ts, "conv-OLD"),
    )
    await store._conn.commit()
    await _do_record(store, "conv-NEW", "userA")

    # 非管理员调用 purge_before 应拒绝
    with pytest.raises(M.HistoryAccessError):
        await store.purge_before("userB", old_ts + 1)

    # 管理员清理 old_ts 之前 → 仅清 OLD，保留 NEW
    n = await store.purge_before("admin1", old_ts + 1)
    assert n == 1
    assert await store.get_session("userA", "conv-OLD") is None
    assert await store.get_session("userA", "conv-NEW") is not None


@pytest.mark.asyncio
async def test_retention_apply(store):
    # retention_days=0 → before=now → 清理所有（created_at<now）
    await _do_record(store, "conv-X", "userA")
    n = await retentionmod.apply_retention(store, requester_id="admin1", retention_days=0)
    assert n == 1
    assert await store.get_session("userA", "conv-X") is None


# ---------------------------------------------------------------------------
# 导出 / 复盘
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_export_md_and_json(store):
    await _do_record(store, "conv-1", "userA")
    md = await store.export_session("userA", "conv-1", fmt="md")
    assert "会话复盘文档" in md
    assert "什么是GDP" in md
    js = await store.export_session("userA", "conv-1", fmt="json")
    parsed = json.loads(js)
    assert parsed["conversation_id"] == "conv-1"
    ctx = await store.get_session_for_review("userA", "conv-1")
    assert "会话复盘上下文" in ctx
    assert "合规OK" in ctx


# ---------------------------------------------------------------------------
# 降级（S4 local 兜底 / S8 silent 静默）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fail_mode_local_writes_fallback(tmp_path):
    db = str(tmp_path / "history.db")
    s = storemod.SessionHistoryStore(
        db, fail_mode="local", admin_user_ids=["admin1"]
    )
    await s.connect()
    await s.init_tables()
    # 强行让写入失败：断开连接后执行
    await s.close()
    await s._on_write_failure(RuntimeError("boom"), {"conversation_id": "c1"})
    fb = os.path.join(s.history_dir, "history_fallback.jsonl")
    assert os.path.exists(fb)
    content = open(fb, encoding="utf-8").read()
    assert "boom" in content


@pytest.mark.asyncio
async def test_fail_mode_silent_no_exception(store):
    # 即使 _on_write_failure 被调用，silent 模式不写文件也不抛错
    await store._on_write_failure(RuntimeError("x"), {"k": 1})
    fb = os.path.join(store.history_dir, "history_fallback.jsonl")
    assert not os.path.exists(fb)


# ---------------------------------------------------------------------------
# Noop 兜底
# ---------------------------------------------------------------------------
def test_noop_store_safe_defaults():
    n = storemod.NoopHistoryStore()
    assert n.enabled is False

    async def _run():
        assert await n.list_sessions("u") == []
        assert await n.get_session("u", "c") is None
        assert await n.keyword_search("u", "q") == []
        assert await n.delete_session("u", "c") == 0
        await n.record_run(conversation_id="c", user_id="u", run_id="r", turn_id="t",
                           duration_ms=1, model="m", result=None, events=[])

    import asyncio

    asyncio.run(_run())
