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
from app.agent.board import ExpertBoard
from app.agent.model_bridge import AgentarModel, extract_content
from app.agent.rag_scope import scoped
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
    # 07 落库用：多智能体细粒度步骤（路由/RAG/专家意见/合成/回写）的结构化记录，
    # 每个元素 {agent, type, content, meta}；由 07 的 build_events(extra_events=...) 落库
    # 到同一 conversation_id 的轨迹（评审 B1 / v2 完整回放）。
    agent_trace: list[dict] = field(default_factory=list)


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
    if not text:
        return None
    # 多领域问题：协调者输出 "Multi" → 走圆桌会商（档B）。
    if text.strip().lower() == "multi":
        return "Multi"
    for scene in SCENES:
        if scene.lower() in text.lower():
            return scene
    return None


# 各领域的跨域关键词提示（用于档A 专家互询的触发判定）。
_DOMAIN_HINTS: dict[str, list[str]] = {
    "Banking": ["存款", "贷款", "银行", "理财", "活期", "定期", "储蓄"],
    "Securities": ["股票", "证券", "A股", "港股", "打新", "ETF", "开户"],
    "Insurance": ["保险", "保单", "理赔", "寿险", "重疾", "年金"],
    "Trust": ["信托", "家族信托", "受托", "受益人"],
    "MutualFunds": ["基金", "公募", "私募", "净值", "申购", "赎回"],
}


def _detect_peer(scene: str, message: str) -> Optional[str]:
    """档A 专家互询：从问题中识别一个**不同于当前场景**的领域关键词，返回其场景名。

    仅在自动路由（scene 由协调者判定）时启用，显式指定 scene 不做跨域改写。
    """
    msg = message or ""
    for other, hints in _DOMAIN_HINTS.items():
        if other == scene:
            continue
        if any(h in msg for h in hints):
            return other
    return None


def _format_opinions(opinions: list[tuple[str, str]]) -> str:
    """把圆桌会商的多专家意见格式化为可读文本。"""
    return "\n".join(f"【{name}】{text}" for name, text in opinions)


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
    user_id: Optional[int] = None,
    use_personal_docs: bool = False,
) -> AgentResult:
    """多智能体流水线主入口。

    Args:
        message (`str`): 用户输入。
        scene (`str | None`): 业务场景；为 None 时由 Coordinator 判定。
        structured (`bool`): 是否附带结构化分析（intent / slots / tool_plan / expression）。
        user_id (`int | None`): 08-个人文档的归属用户；仅用于检索作用域，不进提示词。
        use_personal_docs (`bool`): 是否把该用户的个人文档并入 RAG 召回。

    Returns:
        `AgentResult`: 含最终回复、建议式合规/风险结论、可选结构化字段。

    Raises:
        ModelInvokeError: 模型调用失败（key 缺失 / 网络错误 / 超时）时上抛，非静默。
    """
    try:
        sys = get_system()
        board = ExpertBoard(sys.agents)
        scene_explicit = scene is not None
        # 07 落库：多智能体细粒度步骤（评审 B1 / v2 完整回放）。
        trace: list[dict] = []

        # 1) 路由：未给定 scene 时由 Coordinator 判定；多领域问题返回 "Multi"。
        if scene is None:
            route_msg = Msg(
                name="user",
                content=f"用户问题：{message}\n请判断场景"
                f"（{'/'.join(SCENES)}）。若问题明显涉及多个领域，请只回复 Multi；"
                f"否则只回复单个场景名。",
                role="user",
            )
            scene_text = await _call_agent(sys.agents["coordinator"], route_msg)
            scene = _parse_scene(scene_text) or "Banking"
            trace.append(
                {"agent": "Coordinator", "type": "route",
                 "content": f"场景判定：{scene}", "meta": {"scene": scene, "scene_explicit": scene_explicit}}
            )

        # 2) RAG 检索（经 RAG ReActAgent 的 retrieve 工具 → 06 知识库 + 08 个人文档）
        #    检索作用域用 contextvars 下传，工具签名保持只有 query（防 LLM 编造 user_id）。
        rag_msg = Msg(
            name="user",
            content=f"请检索与以下问题相关的资料：{message}",
            role="user",
        )
        with scoped(user_id=user_id, use_personal_docs=use_personal_docs):
            context = await _call_agent(sys.agents["rag"], rag_msg)
            trace.append(
                {"agent": "rag", "type": "rag", "content": context, "meta": {"scene": scene}}
            )

            # 3) 领域专家作答：档B 圆桌会商 / 档A 单专家 + 同级互询
            if scene == "Multi" or scene not in SCENES:
                # 档B 圆桌会商：多位专家轮流发言，再由 coordinator 合成统一答案。
                base = Msg(
                    name="user",
                    content=f"参考上下文：\n{context}\n\n用户问题：{message}",
                    role="user",
                )
                opinions = await board.roundtable(SCENES, base)
                for name, text in opinions:
                    trace.append(
                        {"agent": name, "type": "expert_opinion", "content": text, "meta": {"scene": scene}}
                    )
                synth_prompt = (
                    f"用户问题：{message}\n\n参考上下文：\n{context}\n\n"
                    "以下为各位领域专家的独立意见，请综合为一份统一、不矛盾的答复：\n"
                    + _format_opinions(opinions)
                )
                draft = await _call_agent(
                    sys.agents["coordinator"],
                    Msg(name="user", content=synth_prompt, role="user"),
                )
                trace.append(
                    {"agent": "Coordinator", "type": "synthesize", "content": draft, "meta": {"scene": scene}}
                )
                revise_role = sys.agents["coordinator"]
            else:
                expert = sys.agents[scene]
                # 档A 专家互询：仅自动路由且检测到跨域关键词时，编排层代专家咨询一个
                # 相关同级，并将其意见并入作答上下文（专家自身仍只调一次）。
                peer = None if scene_explicit else _detect_peer(scene, message)
                if peer:
                    peer_msg = Msg(
                        name="user",
                        content=f"参考上下文：\n{context}\n\n子问题：{message}",
                        role="user",
                    )
                    peer_opinion = await board.consult(peer, peer_msg)
                    trace.append(
                        {"agent": peer, "type": "expert_opinion", "content": peer_opinion,
                         "meta": {"scene": scene, "consulted_by": scene}}
                    )
                    expert_prompt = (
                        f"参考上下文：\n{context}\n\n"
                        f"同级专家（{peer}）的补充意见：\n{peer_opinion}\n\n"
                        f"用户问题：{message}"
                    )
                else:
                    expert_prompt = f"参考上下文：\n{context}\n\n用户问题：{message}"
                draft = await _call_agent(
                    expert, Msg(name="user", content=expert_prompt, role="user")
                )
                trace.append(
                    {"agent": scene, "type": "expert_opinion", "content": draft, "meta": {"scene": scene}}
                )
                revise_role = expert

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

            # 6) 回写环（FR-EX3）：把合规 / 风控结论回灌给作答角色改写一版，
            #    最终 reply 吸收合理修正，同时保留 compliance_notes / risk_flags。
            #    提示词刻意避开「风控 / 合规审核」等子串，避免与分支判定撞车。
            revise_prompt = (
                f"原问题：{message}\n\n当前答案：{draft}\n\n"
                f"审查反馈：{compliance_notes[0]}\n风险反馈：{risk_flags[0]}\n\n"
                "请根据以上审查反馈修订答案，吸收合理修正，保留事实与引用，"
                "不要新增无关内容。"
            )
            draft = await _call_agent(
                revise_role, Msg(name="user", content=revise_prompt, role="user")
            )
            trace.append(
                {"agent": revise_role.name, "type": "revise", "content": draft, "meta": {"scene": scene}}
            )

        # 7) 结构化分析（可选，由 Coordinator 抽取）
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
            agent_trace=trace,
        )
    except ModelInvokeError as exc:
        # 评审 N2：转化为框架级错误并上抛，不得静默吞掉。
        raise ModelInvokeError(f"[02 多智能体] 模型调用失败: {exc}") from exc
