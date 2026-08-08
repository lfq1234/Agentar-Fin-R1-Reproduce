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
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

import agentscope
from agentscope.message import Msg

from app.agent.agents import SCENES, build_agents
from app.agent.board import ExpertBoard, MAX_CONSULT_ROUNDS, _MODEL_LOCK
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
    """调用 agent 并提取文本。

    模型推理（``LocalTransformerModel.generate``）是同步阻塞调用，位于 AgentScope 的
    同步 ``agent.__call__`` 内。用 ``asyncio.to_thread`` 把整次应答搬离 uvicorn 事件循环，
    使服务期间事件循环不被占住；并以 ``_MODEL_LOCK``（与 ``board._call_agent`` 共用）串行化，
    避免并发请求同时调用共享单例 agent 导致对话记忆串味。模型路径（01 ``app/model``）不变。
    """
    async with _MODEL_LOCK:
        out = await asyncio.to_thread(agent, msg)
    if asyncio.iscoroutine(out):
        out = await out
    return extract_content(out)


def _parse_scene(text: str, candidates: Optional[list[str]] = None) -> Optional[str]:
    if not text:
        return None
    lowered = text.strip().lower()
    # 多领域问题：协调者输出 "Multi" → 走圆桌会商（档B）。
    if lowered == "multi":
        return "Multi"
    pool = candidates if candidates else SCENES
    for scene in pool:
        if scene.lower() in lowered:
            return scene
    return None


def _clear_agent_memory(agent) -> None:
    """清空 agent 的对话记忆（保留 sys_prompt 在 memory 外的引用）。

    用在 synthesize / revise 调用 Coordinator 之前——Coordinator 的路由阶段
    刚输出过 "Multi"，若不清记忆，后续综合/回写阶段模型会沿袭 "Multi" 而非
    真正综合专家意见。
    """
    mem = agent.memory
    history = mem.get_memory()
    # AgentScope 的 get_memory() 返回副本，不能用 while history 配合 delete(-1)
    # 否则副本永不为空 → 死循环阻塞事件循环（已验证：海量 WARNING "Skip delete operation"）。
    # 正确做法：取一次长度，按次数删。
    for _ in range(len(history)):
        mem.delete(-1)


# 各领域的跨域关键词提示（仅作预筛，不再做硬路由；路由权交 Coordinator 自主决策）。
_DOMAIN_HINTS: dict[str, list[str]] = {
    "Banking": ["存款", "贷款", "银行", "理财", "活期", "定期", "储蓄", "利率", "利息", "lpr"],
    "Securities": ["股票", "证券", "A股", "港股", "打新", "ETF", "开户"],
    "Insurance": ["保险", "保单", "理赔", "寿险", "重疾", "年金"],
    "Trust": ["信托", "家族信托", "受托", "受益人"],
    "MutualFunds": ["基金", "公募", "私募", "净值", "申购", "赎回"],
}


def _matched_scenes(message: str) -> list[str]:
    """根据关键词判定问题涉及哪些领域；空列表表示通用/非金融问题。"""
    msg = message or ""
    return [scene for scene in SCENES if any(h in msg for h in _DOMAIN_HINTS.get(scene, []))]


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


def _strip_json_fence(text: str) -> str:
    """剥掉模型常带的 JSON 围栏 / 裸 JSON 对象，转为可读文字。"""
    if not text:
        return ""
    s = text.strip()
    # 1) 匹配 ```json ... ``` 围栏
    m = re.search(r"```(?:json|JSON)?\s*\n?(.*?)```", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    # 2) 尝试解析纯 JSON 对象/数组并扁平化
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            parts = []
            for item in parsed:
                if isinstance(item, dict):
                    for k, v in item.items():
                        parts.append(f"{k}: {v}")
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else text
        if isinstance(parsed, dict):
            # 如果是 structured intent 格式，取 expression 字段
            expr = parsed.get("expression", "")
            if expr and isinstance(expr, str) and len(expr) > 3:
                return expr
            return "\n".join(f"{k}: {v}" for k, v in parsed.items()) or text
    except Exception:
        pass
    return s


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

        # 1) 路由：所有消息交 Coordinator agent 自主判定，不做关键词硬匹配。
        #    Coordinator 原生理解"用户和多个智能体聊天"的全局语境，
        #    按语义与上下文决定：Direct（闲聊/问候）| 场景名 | Multi（跨领域/用户要求）。
        if scene is None:
            route_msg = Msg(
                name="user",
                content=(
                    f"用户向你们的团队发来消息：“{message}”\n\n"
                    "作为多智能体团队的协调者，请判断这个问题应该由谁来回答：\n"
                    "- 若与金融完全无关（比如纯闲聊、问候、开玩笑），回复 Direct\n"
                    "- 若涉及特定金融领域，回复对应场景名："
                    f"{'/'.join(SCENES)}\n"
                    "- 若涉及跨领域讨论，或用户（user 角色）在前面说了「想和各位专家聊聊」"
                    "「看看你们团队怎么协作」等想看多人讨论的话，回复 Multi\n"
                    "请只回复一个词（Direct / 场景名 / Multi），不要加解释或标点。"
                ),
                role="user",
            )
            route_text = (await _call_agent(sys.agents["coordinator"], route_msg)).strip()
            # 清理模型可能多余输出的标点与空格
            route_text = route_text.rstrip(".,;!，。；！").strip()
            if route_text == "Multi":
                scene = "Multi"
            elif route_text in SCENES:
                scene = route_text
            else:
                scene = "Direct"  # 失配退化
            trace.append(
                {"agent": "Coordinator", "type": "route",
                 "content": f"场景判定：{scene}（原始：{route_text[:60]}）",
                 "meta": {"scene": scene, "scene_explicit": scene_explicit,
                          "coordinator_raw": route_text}}
            )

        # 1.5) 关键词兜底：Qwen3-0.6B 太小，Coordinator 可能把跨领域金融问题误判为 Direct。
        #    若 Coordinator 说 Direct 但消息命中了金融领域关键词，忽略 Coordinator，按关键词路由。
        if scene == "Direct":
            fallback = _matched_scenes(message)
            if fallback:
                scene = fallback[0] if len(fallback) == 1 else "Multi"
                trace.append(
                    {"agent": "Coordinator", "type": "route",
                     "content": f"场景修正：{scene}（Coordinator 误判 Direct，关键词命中 {fallback}）",
                     "meta": {"scene": scene, "scene_explicit": False,
                              "coordinator_raw": route_text, "keyword_fallback": fallback}}
                )
            else:
                # 真正非金融 / 通用问题由 Direct 助手自然语言回答
                direct_msg = Msg(name="user", content=message, role="user")
                draft = await _call_agent(sys.agents["direct"], direct_msg)
                return AgentResult(reply=draft, agent_trace=[])

        # 2) RAG 检索（经 RAG ReActAgent 的 retrieve 工具 → 06 知识库 + 08 个人文档）
        #    检索作用域用 contextvars 下传，工具签名保持只有 query（防 LLM 编造 user_id）。
        rag_msg = Msg(
            name="user",
            content=f"请检索与以下问题相关的资料：{message}",
            role="user",
        )
        with scoped(user_id=user_id, use_personal_docs=use_personal_docs):
            context = await _call_agent(sys.agents["rag"], rag_msg)
            # RAG 结果仅作内部上下文，不展示给用户。

            # 3) 领域专家作答：档B 圆桌会商按关键词筛选命中领域的专家；档A 单专家直接作答。
            if scene == "Multi":
                # 档B 圆桌会商：按 _matched_scenes 关键词筛选命中领域专家；
                # 若未命中任何关键词则召集全部专家（避免漏答），上限 MAX_CONSULT_ROUNDS。
                target_scenes = _matched_scenes(message) or SCENES
                base = Msg(
                    name="user",
                    content=f"参考上下文：\n{context}\n\n用户问题：{message}",
                    role="user",
                )
                opinions = await board.roundtable(target_scenes, base)
                for name, text in opinions:
                    trace.append(
                        {"agent": name, "type": "expert_opinion", "content": text, "meta": {"scene": scene}}
                    )
                synth_prompt = (
                    f"用户问题：{message}\n\n参考上下文：\n{context}\n\n"
                    "以下为相关领域专家的独立意见，请综合为一份统一、不矛盾的答复：\n"
                    + _format_opinions(opinions)
                )
                # 清空 Coordinator 记忆（防止路由阶段 "Multi" 污染综合输出）。
                _clear_agent_memory(sys.agents["coordinator"])
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
                expert_prompt = f"参考上下文：\n{context}\n\n用户问题：{message}"
                draft = await _call_agent(
                    expert, Msg(name="user", content=expert_prompt, role="user")
                )
                trace.append(
                    {"agent": expert.name, "type": "expert_opinion", "content": draft, "meta": {"scene": scene}}
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
            #    提示词刻意避开「风控 / 合规审核」等子串，避免与分支判定撞车；
            #    明确要求自然语言输出，避免小模型模仿审查/风控的 JSON 风格。
            revise_prompt = (
                f"原问题：{message}\n\n当前答案：{draft}\n\n"
                f"审查反馈：{compliance_notes[0]}\n风险反馈：{risk_flags[0]}\n\n"
                "请根据以上反馈修订答案：用一段自然、通顺的中文回答用户，"
                "吸收合理修正，保留事实，不要新增无关内容。"
                "直接给出修订后的正文，不要输出列表 / JSON / Markdown 代码块。"
            )
            # Multi 通道下 revise_role==Coordinator，清记忆避免合成阶段残留污染回写。
            if revise_role is sys.agents["coordinator"]:
                _clear_agent_memory(revise_role)
            draft = await _call_agent(
                revise_role, Msg(name="user", content=revise_prompt, role="user")
            )
            draft = _strip_json_fence(draft)
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


# 流式事件中 _agent_meta 的共享逻辑（chat_service 中的 A lookalike）
_AGENT_META_SSE: dict[str, tuple[str, str]] = {
    "Coordinator": ("🤖", "协调者"),
    "Direct": ("🤖", "Agentar"),
    "BankingExpert": ("🏦", "银行专家"),
    "SecuritiesExpert": ("📈", "证券专家"),
    "InsuranceExpert": ("🛡️", "保险专家"),
    "TrustExpert": ("🏛️", "信托专家"),
    "MutualFundsExpert": ("🧺", "基金专家"),
    "ComplianceReviewer": ("✅", "合规审核"),
    "RiskController": ("⚠️", "风控"),
}
_AGENT_MENTIONS: dict[str, str] = {
    "BankingExpert": "基金专家",
    "SecuritiesExpert": "保险专家",
    "InsuranceExpert": "信托专家",
    "TrustExpert": "基金专家",
    "MutualFundsExpert": "合规审核",
    "ComplianceReviewer": "风控",
    "RiskController": "协调者",
}


async def run_stream(
    message: str,
    scene: Optional[str] = None,
    user_id: Optional[int] = None,
    use_personal_docs: bool = False,
):
    """多智能体流式版本：每完成一步就 yield 一个 SSE 事件 dict，前端逐条渲染。"""
    sys = get_system()
    board = ExpertBoard(sys.agents)

    # 1) 路由
    if scene is None:
        route_msg = Msg(
            name="user",
            content=(
                f"用户向你们的团队发来消息：\"{message}\"\n\n"
                "作为多智能体团队的协调者，请判断这个问题应该由谁来回答：\n"
                "- 若与金融完全无关（比如纯闲聊、问候、开玩笑），回复 Direct\n"
                "- 若涉及特定金融领域，回复对应场景名："
                f"{'/'.join(SCENES)}\n"
                "- 若涉及跨领域讨论，回复 Multi\n"
                "请只回复一个词（Direct / 场景名 / Multi），不要加解释或标点。"
            ),
            role="user",
        )
        route_text = (await _call_agent(sys.agents["coordinator"], route_msg)).strip()
        route_text = route_text.rstrip(".,;!，。；！").strip()
        if route_text == "Multi":
            scene = "Multi"
        elif route_text in SCENES:
            scene = route_text
        else:
            scene = "Direct"
        # 关键词兜底
        if scene == "Direct":
            fallback = _matched_scenes(message)
            if fallback:
                scene = fallback[0] if len(fallback) == 1 else "Multi"
        yield {"type": "route", "agent": "Coordinator", "scene": scene}

    # 2) Direct 快速通道
    if scene == "Direct":
        yield {"type": "agent_start", "agent": "Direct", **dict(zip(("avatar", "name"), _AGENT_META_SSE["Direct"]))}
        direct_msg = Msg(name="user", content=message, role="user")
        draft = await _call_agent(sys.agents["direct"], direct_msg)
        yield {"type": "agent_message", "agent": "Direct", "content": draft}
        yield {"type": "done", "reply": draft, "conversation_id": None}
        return

    # 3) RAG 检索（内部步骤，不暴露给前端）
    rag_msg = Msg(name="user", content=f"请检索与以下问题相关的资料：{message}", role="user")
    with scoped(user_id=user_id, use_personal_docs=use_personal_docs):
        context = await _call_agent(sys.agents["rag"], rag_msg)

        # 4) 领域专家作答
        if scene == "Multi":
            target_scenes = _matched_scenes(message) or SCENES
            opinions = await board.roundtable(target_scenes[:MAX_CONSULT_ROUNDS], Msg(
                name="user", content=f"参考上下文：\n{context}\n\n用户问题：{message}", role="user"))
            for name, text in opinions:
                yield {"type": "agent_start", "agent": name,
                       **dict(zip(("avatar", "name"), _AGENT_META_SSE.get(name, ("🤖", name))))}
                yield {"type": "agent_message", "agent": name, "content": text,
                       "mention": _AGENT_MENTIONS.get(name)}
            synth_prompt = (
                f"用户问题：{message}\n\n参考上下文：\n{context}\n\n"
                "以下为相关领域专家的独立意见，请综合为一份统一、不矛盾的答复：\n"
                + "\n".join(f"【{n}】{t}" for n, t in opinions)
            )
            _clear_agent_memory(sys.agents["coordinator"])
            yield {"type": "agent_start", "agent": "Coordinator",
                   **dict(zip(("avatar", "name"), _AGENT_META_SSE["Coordinator"]))}
            draft = await _call_agent(sys.agents["coordinator"], Msg(name="user", content=synth_prompt, role="user"))
            yield {"type": "agent_message", "agent": "Coordinator", "content": draft}
            revise_role = sys.agents["coordinator"]
        else:
            expert = sys.agents[scene]
            agent_key = expert.name
            yield {"type": "agent_start", "agent": agent_key,
                   **dict(zip(("avatar", "name"), _AGENT_META_SSE.get(agent_key, ("🤖", agent_key))))}
            expert_prompt = f"参考上下文：\n{context}\n\n用户问题：{message}"
            draft = await _call_agent(expert, Msg(name="user", content=expert_prompt, role="user"))
            yield {"type": "agent_message", "agent": agent_key, "content": draft,
                   "mention": _AGENT_MENTIONS.get(agent_key)}
            revise_role = expert

        # 5) 合规审核
        review_text = await _call_agent(sys.agents["review"], Msg(
            name="user", content=f"原问题：{message}\n待审答案：{draft}", role="user"))
        yield {"type": "agent_start", "agent": "ComplianceReviewer",
               **dict(zip(("avatar", "name"), _AGENT_META_SSE["ComplianceReviewer"]))}
        yield {"type": "agent_message", "agent": "ComplianceReviewer", "content": review_text,
               "mention": "风控"}

        # 6) 风控
        risk_text = await _call_agent(sys.agents["risk"], Msg(
            name="user", content=f"原问题：{message}\n待审答案：{draft}", role="user"))
        yield {"type": "agent_start", "agent": "RiskController",
               **dict(zip(("avatar", "name"), _AGENT_META_SSE["RiskController"]))}
        yield {"type": "agent_message", "agent": "RiskController", "content": risk_text,
               "mention": "协调者"}

        # 7) 回写修订
        if revise_role is sys.agents["coordinator"]:
            _clear_agent_memory(revise_role)
        draft = await _call_agent(revise_role, Msg(
            name="user",
            content=f"原问题：{message}\n\n当前答案：{draft}\n\n审查反馈：{review_text}\n风险反馈：{risk_text}\n\n"
                    "请根据以上反馈修订答案：用一段自然、通顺的中文回答用户，吸收合理修正，保留事实。"
                    "直接给出修订后的正文，不要输出列表 / JSON / Markdown 代码块。",
            role="user"))
        draft = _strip_json_fence(draft)
        yield {"type": "agent_start", "agent": "Coordinator",
               **dict(zip(("avatar", "name"), _AGENT_META_SSE["Coordinator"]))}
        yield {"type": "agent_message", "agent": "Coordinator", "content": draft}

    yield {"type": "done", "reply": draft, "conversation_id": None}
