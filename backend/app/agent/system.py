"""02-多智能体基础框架：运行时初始化 + 流水线编排 + 统一入口。

`run()` 是框架对外的主入口：
    路由(Coordinator) → RAG 检索(占位) → 领域专家作答 → 审核(建议式) → 风控(建议式)
    → 结构化分析(可选) → AgentResult

所有 LLM 调用均经 01 的 `get_model().generate()`（评审 G1 · 方案 X）。
`ModelInvokeError` 由 01 抛出后在此捕获并转化为框架级错误（评审 N2，非静默）。
"""
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Optional

import agentscope
from agentscope.message import Msg

from app.agent.agents import SCENES, build_agents
from app.agent.model_bridge import AgentarModel, extract_content
from app.model import get_model
from app.model.exceptions import ModelInvokeError

# agentscope.init 注册的占位模型配置（仅供 ReActAgent/DialogAgent 子类 __init__ 时不崩，
# 真正的模型执行由 AgentarModel 覆盖 self.model 后走 01 的 generate()）。
PLACEHOLDER_CONFIG_NAME = "agentar-placeholder"
_PLACEHOLDER_CONFIG = {
    "model_type": "openai_chat",
    "config_name": PLACEHOLDER_CONFIG_NAME,
    "api_key": "EMPTY",
    "base_url": "http://localhost:8000/v1",
    "generate_args": {"temperature": 0.3, "stream": False},
}

_STRUCT_PROMPT = (
    "请基于以下用户问题与专家答案，抽取结构化分析，严格只输出一个 JSON 对象，"
    "字段为 intent / slots / tool_plan / expression，不要输出多余文字。\n"
    "用户问题：{message}\n专家答案：{draft}"
)


@dataclass
class AgentResult:
    """多智能体流水线的统一返回结构。"""

    reply: str = ""                          # 领域专家的主回复（最终输出）
    compliance_notes: list[str] = field(default_factory=list)  # 审核：合规提示（建议式）
    risk_flags: list[str] = field(default_factory=list)        # 风控：风险警示（建议式）
    intent: Optional[str] = None             # analyze 模式
    slots: dict[str, str] = field(default_factory=dict)
    tool_plan: list[str] = field(default_factory=list)
    expression: str = ""                      # 给前端的表达 / 话术


class _AgentSystem:
    """多智能体系统（懒加载、进程内有且仅有一份）。"""

    def __init__(self, agents: dict) -> None:
        self.agents = agents


_lock = threading.Lock()
_system: Optional[_AgentSystem] = None


def _ensure_agentscope_init() -> None:
    """注册占位模型配置，使 ReActAgent/DialogAgent 子类 __init__ 能解析 model_config_name。"""
    agentscope.init(model_configs=[_PLACEHOLDER_CONFIG])


def get_system() -> _AgentSystem:
    """获取（懒加载、线程安全）多智能体系统单例。"""
    global _system
    if _system is None:
        with _lock:
            if _system is None:
                _ensure_agentscope_init()
                _system = _AgentSystem(build_agents(get_model()))
    return _system


async def _call_agent(agent, msg: Msg) -> str:
    """调用 agent 并提取文本。ReActAgent 为 async，DialogAgent 为 sync，统一处理。"""
    out = agent(msg)
    if asyncio.iscoroutine(out):
        out = await out
    return extract_content(out)


def _parse_scene(text: str) -> Optional[str]:
    for scene in SCENES:
        if scene.lower() in text.lower():
            return scene
    return None


def _parse_structured(text: str) -> dict:
    """从模型输出解析结构化 JSON；失败则优雅降级为空结构。"""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        data = json.loads(text[start : end + 1])
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, ValueError):
        return {}


async def run(
    message: str,
    scene: Optional[str] = None,
    structured: bool = False,
) -> AgentResult:
    """多智能体流水线主入口。

    Args:
        message (`str`): 用户输入。
        scene (`str | None`): 业务场景；为 None 时由 Coordinator 判定。
        structured (`bool`): 是否附带结构化分析（intent / slots / tool_plan / expression）。

    Returns:
        `AgentResult`: 含最终回复、建议式合规/风险结论、可选结构化字段。

    Raises:
        ModelInvokeError: 模型调用失败（key 缺失 / 网络错误 / 超时）时上抛，非静默。
    """
    try:
        sys = get_system()

        # 1) 路由：未给定 scene 时由 Coordinator 判定
        if scene is None:
            route_msg = Msg(
                name="user",
                content=f"用户问题：{message}\n请判断场景"
                f"（{'/'.join(SCENES)}），只回复场景名。",
                role="user",
            )
            scene_text = await _call_agent(sys.agents["coordinator"], route_msg)
            scene = _parse_scene(scene_text) or "Banking"

        # 2) RAG 检索（经 RAG ReActAgent 的 retrieve 工具，本期返回占位上下文）
        rag_msg = Msg(
            name="user",
            content=f"请检索与以下问题相关的资料：{message}",
            role="user",
        )
        context = await _call_agent(sys.agents["rag"], rag_msg)

        # 3) 领域专家作答（消费检索上下文）
        expert = sys.agents[scene]
        expert_msg = Msg(
            name="user",
            content=f"参考上下文：\n{context}\n\n用户问题：{message}",
            role="user",
        )
        draft = await _call_agent(expert, expert_msg)

        # 4) 审核（建议式，不阻断）
        review_msg = Msg(
            name="user",
            content=f"原问题：{message}\n待审答案：{draft}",
            role="user",
        )
        compliance_notes = [await _call_agent(sys.agents["review"], review_msg)]

        # 5) 风控（建议式，不阻断）
        risk_msg = Msg(
            name="user",
            content=f"原问题：{message}\n待审答案：{draft}",
            role="user",
        )
        risk_flags = [await _call_agent(sys.agents["risk"], risk_msg)]

        # 6) 结构化分析（可选，由 Coordinator 抽取）
        intent = None
        slots: dict[str, str] = {}
        tool_plan: list[str] = []
        expression = ""
        if structured:
            sa_msg = Msg(
                name="user",
                content=_STRUCT_PROMPT.format(message=message, draft=draft),
                role="user",
            )
            parsed = _parse_structured(await _call_agent(sys.agents["coordinator"], sa_msg))
            intent = parsed.get("intent")
            raw_slots = parsed.get("slots", {})
            slots = {str(k): str(v) for k, v in raw_slots.items()} if isinstance(raw_slots, dict) else {}
            tool_plan = parsed.get("tool_plan", []) or []
            expression = parsed.get("expression", "") or ""

        return AgentResult(
            reply=draft,
            compliance_notes=compliance_notes,
            risk_flags=risk_flags,
            intent=intent,
            slots=slots,
            tool_plan=tool_plan if isinstance(tool_plan, list) else [],
            expression=expression,
        )
    except ModelInvokeError as exc:
        # 评审 N2：转化为框架级错误并上抛，不得静默吞掉。
        raise ModelInvokeError(f"[02 多智能体] 模型调用失败: {exc}") from exc
