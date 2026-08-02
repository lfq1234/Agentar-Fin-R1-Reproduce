"""02-多智能体基础框架：测试。

用 fake 模型（实现 01 的 ModelInterface.generate()）替代真实 LLM，验证：
- 模型调用确实经由 01 的 generate()（评审 G1）；
- 场景路由命中对应专家、Coordinator 能判定场景；
- stub 检索下流水线端到端跑通；
- 结构化 JSON 可解析、非法时优雅降级；
- ModelInvokeError 透传（非静默，评审 N2）；
- 9 个角色齐备、model.mode 切换对框架透明。

无 OPENAI_API_KEY / 无 vLLM 也能跑（全程 fake 模型）。
"""
from __future__ import annotations

import json

import pytest

from app.agent.agents import SCENES, build_agents
from app.agent.system import AgentResult, get_system, run
from app.model import ModelInterface
from app.model.exceptions import ModelInvokeError


class FakeModel(ModelInterface):
    """记录 generate 调用次数，并按 prompt 内容返回可解析的 ReAct 回复。

    注意：分支判定用**唯一 token**，避免与专家 prompt 中出现的「风险」等词撞车。
    """

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)
        self.calls: list[str] = []

    def build_agentscope_config(self) -> dict:
        return {}

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        # 结构化分析分支（Coordinator 抽取；prompt 含唯一标记）
        if "严格只输出一个 JSON" in prompt:
            return "Thought: 抽取完成\nFinish: " + json.dumps(
                {"intent": "consult", "slots": {"product": "基金"},
                 "tool_plan": ["lookup"], "expression": "好的，为您解答"}
            )
        # RAG 检索分支
        if "知识检索" in prompt:
            return "Thought: 检索完成\nFinish: [RAG] 暂无检索结果"
        # 合规审核分支（review.txt 含「合规审核」）
        if "合规审核" in prompt:
            return "Thought: 审查完成\nFinish: [Review] 已合规，含风险提示"
        # 风控分支（risk.txt 含「风控」，专家 prompt 仅含「风险」不撞车）
        if "风控" in prompt:
            return "Thought: 评估完成\nFinish: [Risk] 风险等级中，未发现高危"
        # 协调者路由分支（路由调用 user 消息含「请判断场景」）
        if "请判断场景" in prompt:
            return "Thought: 路由\nFinish: Securities"
        # 领域专家分支（prompt 含「领域」+ 场景名）
        for scene in SCENES:
            if scene in prompt and "领域" in prompt:
                return f"Thought: 作答完成\nFinish: [{scene}] 这是{scene}领域答案"
        return "Thought: 完成\nFinish: [fallback] 回复"


class ErrorModel(ModelInterface):
    """generate 直接抛 ModelInvokeError，验证异常透传。"""

    def __init__(self, cfg=None) -> None:
        super().__init__(cfg)

    def build_agentscope_config(self) -> dict:
        return {}

    def generate(self, prompt: str, **kwargs) -> str:
        raise ModelInvokeError("key 缺失 / 网络错误 / 超时（fake）")


@pytest.fixture
def fake_model(monkeypatch):
    """在每个测试前重置系统单例，并把 01 的 get_model 替换为 FakeModel。"""
    import app.agent.system as system_mod

    system_mod._system = None
    fm = FakeModel()
    monkeypatch.setattr(system_mod, "get_model", lambda: fm)
    return fm


def test_nine_roles_present(monkeypatch):
    """9 个角色齐备：Coordinator + 5 专家 + RAG + 审核 + 风控。"""
    import app.agent.system as system_mod

    system_mod._system = None
    fm = FakeModel()
    monkeypatch.setattr(system_mod, "get_model", lambda: fm)
    system_mod._ensure_agentscope_init()  # ReActAgent.__init__ 需先注册占位模型
    agents = build_agents(fm)
    expected = {"coordinator", *SCENES, "rag", "review", "risk"}
    assert set(agents.keys()) == expected
    assert len(agents) == 9


@pytest.mark.asyncio
async def test_model_call_routes_through_generate(fake_model):
    """评审 G1：02 对 LLM 的调用全部经 01 的 generate()。"""
    await run("用户问：怎么买基金", scene="Banking")
    assert len(fake_model.calls) >= 1, "应至少调用一次 01 的 generate()"


@pytest.mark.asyncio
async def test_scene_routing_hits_expert(fake_model):
    """给定 scene 命中对应专家；Coordinator 也能判定场景。"""
    # 显式 scene
    res = await run("用户问：基金怎么买", scene="Insurance")
    assert "[Insurance]" in res.reply, f"应路由到 Insurance 专家，实际: {res.reply}"

    # 隐式 scene：Coordinator 判定为 Securities
    fake_model.calls.clear()
    res2 = await run("用户问：股票开户流程")
    assert "[Securities]" in res2.reply, f"应路由到 Securities，实际: {res2.reply}"


@pytest.mark.asyncio
async def test_pipeline_runs_with_stub_retrieval(fake_model):
    """RAG 检索为占位（空）时，流水线仍能端到端跑通。"""
    res = await run("用户问：存款利率", scene="Banking")
    assert isinstance(res, AgentResult)
    assert res.reply, "reply 不应为空"
    assert res.compliance_notes and "[Review]" in res.compliance_notes[0]
    assert res.risk_flags and "[Risk]" in res.risk_flags[0]


@pytest.mark.asyncio
async def test_structured_parsing(fake_model):
    """structured=True 时产出结构化字段；非法 JSON 优雅降级。"""
    res = await run("用户问：基金定投", scene="MutualFunds", structured=True)
    assert res.intent == "consult"
    assert res.slots.get("product") == "基金"
    assert res.tool_plan == ["lookup"]
    assert res.expression

    # 非法 JSON 降级
    class BrokenFake(FakeModel):
        def generate(self, prompt, **kwargs):
            self.calls.append(prompt)
            return "Finish: 这不是json"

    import app.agent.system as system_mod

    system_mod._system = None
    bf = BrokenFake()
    monkeypatch_patch = pytest.MonkeyPatch()
    monkeypatch_patch.setattr(system_mod, "get_model", lambda: bf)
    try:
        res2 = await run("用户问：理财", scene="Banking", structured=True)
        assert res2.intent is None
        assert res2.slots == {}
    finally:
        monkeypatch_patch.undo()


@pytest.mark.asyncio
async def test_model_invoke_error_propagates(monkeypatch):
    """评审 N2：ModelInvokeError 透传，不静默吞掉。"""
    import app.agent.system as system_mod

    system_mod._system = None
    monkeypatch.setattr(system_mod, "get_model", lambda: ErrorModel())
    with pytest.raises(ModelInvokeError):
        await run("用户问：任意", scene="Banking")
