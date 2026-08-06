"""02-多智能体基础框架：专家互相通讯协作测试（档A 互询 / 档B 会商 / 回写环 / 作用域守恒）。

用极轻量 fake agent（不依赖 agentscope / 真实 LLM / 数据库）验证编排逻辑，
对照需求文档 §8（FR-EX1~EX5）与技术文档 §14。

关键约束（避免与 _reply_fn 的脆弱分支判定撞车）：
- 回写环 revise 提示词不含「风控 / 合规审核」子串（FR-EX3 验证依赖 [REVISED] 标记）；
- 圆桌会商合成提示词含「综合为一份 / 独立意见」（FR-EX2 验证依赖 [SYNTH] 标记）。
"""
from __future__ import annotations

import pytest
from agentscope.message import Msg

from app.agent.board import ExpertBoard, MAX_CONSULT_ROUNDS
from app.agent.rag_scope import current_scope, scoped
from app.agent.system import SCENES, run


class _FakeAgent:
    """极简 agent：记录调用，按 reply_fn 返回 Msg。供编排测试，无需 agentscope。"""

    def __init__(self, name: str, reply_fn) -> None:
        self.name = name
        self._reply_fn = reply_fn
        self.calls: list[Msg] = []

    def __call__(self, msg: Msg) -> Msg:
        self.calls.append(msg)
        return Msg(self.name, self._reply_fn(self.name, msg), "assistant")


# 路由结果由测试动态控制（"Multi" 触发档B）。
_CFG = {"route": "Banking"}

# 每次 agent 调用时记录当前 RAG 作用域，用于验证 FR-EX4（作用域守恒）。
SCOPE_CAPTURES: list = []


def _reply_fn(name: str, msg: Msg) -> str:
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    SCOPE_CAPTURES.append(current_scope())
    if "请判断场景" in content:
        return _CFG["route"]
    if "请检索" in content:
        return "[RAG]"
    if "待审答案" in content:  # 审核 / 风控（建议式）
        return "[NOTE]"
    if "审查反馈" in content or "风险反馈" in content:  # 回写环改写
        return "[REVISED]"
    if "综合为一份" in content or "独立意见" in content:  # 档B 合成
        return "[SYNTH]"
    return f"[{name}-DRAFT]"


def _build_fake_system() -> object:
    """返回带 agents 命名空间的对象，供 monkeypatch get_system 使用。"""
    agents = {
        name: _FakeAgent(name, _reply_fn)
        for name in ["coordinator", "rag", "review", "risk", *SCENES]
    }
    return type("Sys", (), {"agents": agents})()


@pytest.fixture
def fake_system(monkeypatch):
    import app.agent.system as system_mod

    system_mod._system = None
    sys_obj = _build_fake_system()
    monkeypatch.setattr(system_mod, "get_system", lambda: sys_obj)
    yield sys_obj
    _CFG["route"] = "Banking"  # 复位，避免影响其它测试


# ── 档A 专家互询（FR-EX1） ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_board_consult_calls_peer(fake_system):
    """consult 直接调用同级专家并取其文本意见。"""
    board = ExpertBoard(fake_system.agents)
    out = await board.consult("Insurance", Msg("user", "子问题", "user"))
    assert out == "[Insurance-DRAFT]"
    assert len(fake_system.agents["Insurance"].calls) == 1


@pytest.mark.asyncio
async def test_run_peer_consult_on_cross_domain(fake_system):
    """自动路由的跨域问题应触发同级专家互询（FR-EX1）。"""
    _CFG["route"] = "Insurance"  # 路由到保险，但问题含「基金」→ 跨域
    res = await run("保险怎么买，另外基金定投可行吗")
    assert fake_system.agents["MutualFunds"].calls, "跨域应触发同级专家互询"
    assert res.reply == "[REVISED]"  # 经回写环改写


@pytest.mark.asyncio
async def test_run_no_peer_consult_when_single_domain(fake_system):
    """单域问题不触发互询（避免无谓的额外模型调用）。"""
    _CFG["route"] = "Banking"
    await run("银行存款利率多少")
    assert not fake_system.agents["Insurance"].calls
    assert not fake_system.agents["Securities"].calls


# ── 档B 圆桌会商（FR-EX2） ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_board_roundtable_fanout(fake_system):
    """roundtable 顺序调用多位专家并收集意见。"""
    board = ExpertBoard(fake_system.agents)
    opinions = await board.roundtable(["Banking", "Securities"], Msg("user", "q", "user"))
    assert len(opinions) == 2
    assert {name for name, _ in opinions} == {"Banking", "Securities"}


@pytest.mark.asyncio
async def test_roundtable_caps_at_max(fake_system):
    """会商参与专家数受 MAX_CONSULT_ROUNDS 限制（FR-EX5）。"""
    board = ExpertBoard(fake_system.agents)
    opinions = await board.roundtable(list(SCENES), Msg("user", "q", "user"))
    assert len(opinions) == MAX_CONSULT_ROUNDS
    assert MAX_CONSULT_ROUNDS < len(SCENES)


@pytest.mark.asyncio
async def test_run_multi_domain_triggers_roundtable(fake_system):
    """scene=Multi 时触发圆桌会商，≥2 专家被调用并经合成（FR-EX2）。"""
    _CFG["route"] = "Multi"
    res = await run("保险和信托怎么搭配更稳妥")
    consulted = [n for n in SCENES if fake_system.agents[n].calls]
    assert len(consulted) >= 2, "圆桌会商应 fan-out 给多位专家"
    assert res.reply == "[REVISED]"  # 会商合成后进入回写环


# ── 回写环（FR-EX3） ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rewrite_loop_runs(fake_system):
    """审核/风控结论回灌后，最终 reply 改变（FR-EX3）。"""
    _CFG["route"] = "Banking"
    res = await run("存款利率")
    assert res.compliance_notes and res.compliance_notes[0] == "[NOTE]"
    assert res.risk_flags and res.risk_flags[0] == "[NOTE]"
    assert res.reply == "[REVISED]"  # 回写环改写版覆盖初稿


# ── 作用域守恒（FR-EX4） ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scope_preserved_through_consult(fake_system):
    """board.consult 调用链中 ContextVar（RAG 作用域）保持不变（FR-EX4）。

    注：Python 的 __call__ 在类型上查找，实例属性无法覆盖，故改为在 reply_fn
    内记录每次 agent 调用的 current_scope()（见 SCOPE_CAPTURES）。
    """
    SCOPE_CAPTURES.clear()
    board = ExpertBoard(fake_system.agents)
    with scoped(user_id=7, use_personal_docs=True):
        await board.consult("Insurance", Msg("user", "q", "user"))
    assert SCOPE_CAPTURES, "应触发一次子专家调用"
    assert any(s.user_id == 7 and s.use_personal_docs is True for s in SCOPE_CAPTURES)
