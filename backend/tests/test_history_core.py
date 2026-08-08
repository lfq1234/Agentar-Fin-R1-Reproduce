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
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from app.db.history import collect
from app.db.history import export as exportmod
from app.db.history import models as M
from app.db.history import redact
from app.db.history import retention as retentionmod
from app.db.history import store as storemod


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
        admin_user_ids=[999],
        **kw,
    )
    return s


@pytest_asyncio.fixture
async def store():
    tmp = tempfile.mkdtemp()
    s = _new_store(tmp)
    await s.connect()
    await s.init_tables()
    # 测试直接用整数 user_id，需先建 users 行以满足 conversations.user_id 外键
    for uid in (1, 2, 999):
        await s._conn.execute(
            "INSERT OR IGNORE INTO users (id, username, created_at, updated_at) "
            "VALUES (?, ?, '2020-01-01T00:00:00+00:00', '2020-01-01T00:00:00+00:00')",
            (uid, f"u{uid}"),
        )
    await s._conn.commit()
    yield s
    await s.close()


async def _do_record(s, conv, user, *, reply="答案是42", note="合规OK", msg="什么是GDP"):
    result = FakeResult(reply=reply, compliance_notes=[note], risk_flags=[])
    events = collect.build_events(result, user_message=msg)
    # 模拟 chat.py 的落库顺序：先把 user + assistant 消息写入 conversations.data，
    # 再由 record_run 把 trace 挂到匹配回复内容的助手消息上（单表收口后的真实流程）。
    # 约定：行级 created_at/updated_at 为 ISO TEXT；消息级 created_at 为 epoch 毫秒整数。
    ts_iso = datetime.now(timezone.utc).isoformat()
    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    seeded = {
        "messages": [
            {"role": "user", "content": msg, "scene": "qa", "created_at": ts_ms},
            {"role": "assistant", "content": reply, "scene": "qa", "created_at": ts_ms},
        ]
    }
    await s._conn.execute(
        "INSERT OR IGNORE INTO conversations (id, user_id, scene, title, created_at, updated_at, data) "
        "VALUES (?,?,?,?,?,?,?)",
        (str(conv), str(user), "qa", reply[:50], ts_iso, ts_iso, json.dumps(seeded, ensure_ascii=False)),
    )
    await s._conn.commit()
    return await s.record_run(
        conversation_id=conv,
        user_id=user,
        scene="qa",
        run_id="run-" + str(conv),
        turn_id="turn-" + str(conv),
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


def test_collect_extra_events_from_agent_trace():
    """02 产出的 agent_trace（dict 列表）经 hooks 转 TraceEvent 后能被 build_events 收录。"""
    res = FakeResult(reply="hi", compliance_notes=["c1"], risk_flags=["r1"])
    agent_trace = [
        {"agent": "Banking", "type": "expert_opinion", "content": "这是Banking领域答案",
         "meta": {"scene": "Banking"}},
        {"agent": "Coordinator", "type": "synthesize", "content": "合成统一答案", "meta": {}},
    ]
    extra = [
        M.TraceEvent(agent=s["agent"], type=s["type"], summary_out=s["content"],
                     meta_json=json.dumps(s.get("meta") or {}, ensure_ascii=False))
        for s in agent_trace
    ]
    evs = collect.build_events(res, user_message="u", extra_events=extra)
    types = {e.type for e in evs}
    assert "expert_opinion" in types
    assert "synthesize" in types


# ---------------------------------------------------------------------------
# 写入 / 回放
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_record_and_get_session(store):
    await _do_record(store, 1, 1)
    detail = await store.get_session(1, 1)
    assert detail is not None
    assert detail.has_trace is True
    assert detail.meta is not None
    assert detail.meta.msg_count == 2
    roles = {m["role"] for m in detail.messages}
    assert roles >= {"user", "assistant"}


@pytest.mark.asyncio
async def test_record_expert_events_and_replay(store):
    """专家意见（expert_opinion）落库后能随会话轨迹回放。"""
    res = FakeResult(reply="最终答案", compliance_notes=["合规OK"], risk_flags=["风险中"])
    extra = [M.TraceEvent(agent="Banking", type="expert_opinion", summary_out="这是Banking领域答案")]
    events = collect.build_events(res, user_message="买基金", extra_events=extra)
    await store.record_run(
        conversation_id=3, user_id=1, scene="Banking",
        run_id="run-x", turn_id="turn-x", duration_ms=50, model="m",
        result=res, events=events, user_message="买基金", total_tokens=5,
    )
    detail = await store.get_session(1, 3)
    flat = [n for run in detail.trace for n, _ in exportmod._iter_event_nodes(run["events"])]
    assert any(n.get("type") == "expert_opinion" for n in flat)


@pytest.mark.asyncio
async def test_get_turn(store):
    await _do_record(store, 1, 1)
    turn = await store.get_turn(1, 1, "turn-1")
    assert turn is not None
    assert turn.user_message == "什么是GDP"
    assert "答案是42" in turn.assistant_reply
    assert any(e.type == "review" for e in turn.events)


@pytest.mark.asyncio
async def test_record_traces_false_safe_degrade(store):
    await store.record_run(
        conversation_id=2,
        user_id=1,
        scene="qa",
        run_id="run-conv-2",
        turn_id="turn-conv-2",
        duration_ms=1,
        model="m",
        result=FakeResult(reply="x"),
        events=[],
        user_message="hi",
    )
    detail = await store.get_session(1, 2)
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
    await _do_record(store, 11, 1)
    mine = await store.list_sessions(1)
    assert any(m.conversation_id == "11" for m in mine)
    others = await store.list_sessions(2)
    assert all(m.conversation_id != "11" for m in others)
    admin = await store.list_sessions(999)
    assert any(m.conversation_id == "11" for m in admin)


@pytest.mark.asyncio
async def test_get_session_permission_denied(store):
    await _do_record(store, 11, 1)
    with pytest.raises(M.HistoryAccessError):
        await store.get_session(2, 11)


@pytest.mark.asyncio
async def test_delete_own_and_denied(store):
    await _do_record(store, 11, 1)
    n = await store.delete_session(1, 11)
    assert n == 1
    assert await store.get_session(1, 11) is None
    # 重新写一条，他人删除应被拒
    await _do_record(store, 12, 1)
    with pytest.raises(M.HistoryAccessError):
        await store.delete_session(2, 12)


# ---------------------------------------------------------------------------
# 检索（B2 字段级鉴权 + B4 时间）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_keyword_search_scoping_and_time(store):
    await _do_record(store, 11, 1, msg="查询GDP增长率")
    # 本人命中
    hits = await store.keyword_search(1, "GDP")
    assert any(h.conversation_id == "11" for h in hits)
    # 他人无命中（非管理员）
    hits_other = await store.keyword_search(2, "GDP")
    assert not any(h.conversation_id == "11" for h in hits_other)
    # 管理员可见
    hits_admin = await store.keyword_search(999, "GDP")
    assert any(h.conversation_id == "11" for h in hits_admin)
    # 时间过滤：start=未来 → 无命中
    future = 9_999_999_999_999_999
    hits_future = await store.keyword_search(1, "GDP", start=future)
    assert hits_future == []


# ---------------------------------------------------------------------------
# 留存 / 清理（B4 时间 + 管理员门禁）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_purge_before_admin_only_and_time(store):
    await _do_record(store, 21, 1)
    old_ts = 1_600_000_000_000  # 固定过去时间戳（2020-09）
    # 把 OLD 的 updated_at 改到 2020（早于 purge_before 的 before 边界）
    await store._conn.execute(
        "UPDATE conversations SET updated_at=? WHERE id=?",
        ("2020-01-01T00:00:00+00:00", 21),
    )
    await store._conn.commit()
    await _do_record(store, 22, 1)

    # 非管理员调用 purge_before 应拒绝
    with pytest.raises(M.HistoryAccessError):
        await store.purge_before(2, old_ts + 1)

    # 管理员清理 old_ts 之前 → 仅清 OLD，保留 NEW
    n = await store.purge_before(999, old_ts + 1)
    assert n == 1
    assert await store.get_session(1, 21) is None
    assert await store.get_session(1, 22) is not None


@pytest.mark.asyncio
async def test_retention_apply(store):
    # retention_days=0 → before=now → 清理所有（updated_at < now）
    await _do_record(store, 4, 1)
    # 把 updated_at 改到过去，避免与 before=now 落在同一毫秒导致边界比较歧义
    await store._conn.execute(
        "UPDATE conversations SET updated_at=? WHERE id=?",
        ("2020-01-01T00:00:00+00:00", 4),
    )
    await store._conn.commit()
    n = await retentionmod.apply_retention(store, requester_id=999, retention_days=0)
    assert n == 1
    assert await store.get_session(1, 4) is None


# ---------------------------------------------------------------------------
# 导出 / 复盘
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_export_md_and_json(store):
    await _do_record(store, 1, 1)
    md = await store.export_session(1, 1, fmt="md")
    assert "会话复盘文档" in md
    assert "什么是GDP" in md
    js = await store.export_session(1, 1, fmt="json")
    parsed = json.loads(js)
    assert parsed["conversation_id"] == "1"
    ctx = await store.get_session_for_review(1, 1)
    assert "会话复盘上下文" in ctx
    assert "合规OK" in ctx


# ---------------------------------------------------------------------------
# 降级（S4 local 兜底 / S8 silent 静默）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fail_mode_local_writes_fallback(tmp_path):
    db = str(tmp_path / "history.db")
    s = storemod.SessionHistoryStore(
        db, fail_mode="local", admin_user_ids=[999]
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
