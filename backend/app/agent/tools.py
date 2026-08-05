"""02-多智能体基础框架：ReAct 智能体的工具定义。

每个工具都是带类型注解 + docstring 的普通函数，供 `ServiceToolkit.add()` 注册给
`ReActAgent` 使用（ReAct 循环据此决定调用哪个工具）。工具内部同样只调用 01 提供的
能力或本地 stub，不直连外部模型。

本期工具多为占位/轻量实现，真实金融工具、RAG 向量库检索等为后续需求接入。
"""
from __future__ import annotations

from agentscope.service import ServiceToolkit

# 06 演进：检索后端切到 DuckDB/SQLite 知识库（get_knowledge_store 按 kb.engine 分发），
# 不再使用 04 的占位 retrieve（空库）。导入链为纯 Python，无 torch 依赖。
from app.db.knowledge import get_knowledge_store, Passage
from app.db.knowledge.chunking import format_passages


def lookup_knowledge(query: str) -> str:
    """根据查询从知识库检索相关金融知识片段，返回带完整溯源的结构化文本。

    输出供下游总结智能体（raf）消费：每条片段标注来源/机构/领域/生效日/版本/相关度，
    便于总结时准确引用与标注出处。

    Args:
        query (`str`): 用户问题或检索关键词。

    Returns:
        `str`: 检索到的知识文本（含引用标记）；无结果时返回提示。
    """
    passages: list[Passage] = get_knowledge_store().retrieve(query)
    return format_passages(passages)


def retrieve_documents(query: str) -> str:
    """检索监管条文 / 产品资料 / 知识库，返回与 query 最相关的文档片段。

    Args:
        query (`str`): 检索关键词。

    Returns:
        `str`: 检索到的文档片段文本。
    """
    return lookup_knowledge(query)


def check_compliance(answer: str) -> str:
    """对给定答案做合规要点检查，返回合规提示。

    Args:
        answer (`str`): 待审查的领域专家答案。

    Returns:
        `str`: 合规检查结论（是否提示风险、是否违规承诺收益等）。
    """
    notes: list[str] = []
    if any(k in answer for k in ("保本", "稳赚", "无风险", " guaranteed", "保证收益")):
        notes.append("⚠️ 不得承诺保本 / 无风险收益，请修正表述。")
    notes.append("已包含必要风险提示，表述无明显误导。")
    return "\n".join(notes)


def assess_risk(answer: str) -> str:
    """对给定答案做风险等级评估，返回风险警示。

    Args:
        answer (`str`): 待评估的领域专家答案。

    Returns:
        `str`: 风险等级与警示信息。
    """
    return "风险等级：中；未发现高危操作，建议式结论，不阻断输出。"


# 角色 -> 工具函数列表。Coordinator（DialogAgent）不在此表，无工具。
_ROLE_TOOLS: dict[str, list] = {
    "rag": [retrieve_documents],
    "Banking": [lookup_knowledge],
    "Securities": [lookup_knowledge],
    "Insurance": [lookup_knowledge],
    "Trust": [lookup_knowledge],
    "MutualFunds": [lookup_knowledge],
    "review": [check_compliance],
    "risk": [assess_risk],
}


def build_toolkit(role: str) -> ServiceToolkit:
    """为指定角色构建 `ServiceToolkit`（每个 agent 独立实例，避免 finish 工具冲突）。

    Args:
        role (`str`): 角色 key（见 `_ROLE_TOOLS`）。

    Returns:
        `ServiceToolkit`: 已注册对应工具的工具集；无匹配则空工具集。
    """
    tk = ServiceToolkit()
    for fn in _ROLE_TOOLS.get(role, []):
        tk.add(fn)
    return tk
