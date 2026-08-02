"""02-多智能体基础框架：智能体定义与构建。

角色构成（共 9 个，评审 N1 已统一计数）：
- Coordinator（协调者 / 路由 / 意图识别）：`DialogAgent`（同步），驱动流水线。
- 5 个领域专家 Banking / Securities / Insurance / Trust / MutualFunds：`ReActAgent`。
- RAG（知识检索占位）、审核（合规）、风控（风险）：`ReActAgent`。

> 评审结论（用户确认）：专家、审核、风控、RAG 一律使用 `ReActAgent`（带工具）；
> 仅 Coordinator 使用 `DialogAgent` 做路由。

所有 agent 的模型执行都通过 `AgentarModel` 委托给 01 的 `get_model().generate()`
（评审 G1 · 方案 X）。AgentScope 在本框架中只承担编排，禁止任何 agent 直连模型。

注意：`ReActAgent` / `DialogAgent` 的 `super().__init__` 会按 `model_config_name`
从 `agentscope.init` 加载一个占位模型对象，我们随后用 `AgentarModel` 覆盖 `self.model`，
从而把模型执行重定向到 01。占位模型的 `api_key="EMPTY"` 不会触发真实请求。
"""
from __future__ import annotations

import os

from agentscope.agents import DialogAgent, ReActAgent

from app.agent.model_bridge import AgentarModel
from app.agent.tools import build_toolkit
from app.model import ModelInterface

# agentscope.init 注册的占位模型配置名（与 system._PLACEHOLDER_CONFIG 对应）。
PLACEHOLDER_CONFIG_NAME = "agentar-placeholder"

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def _load_prompt(name: str) -> str:
    with open(os.path.join(_PROMPT_DIR, f"{name}.txt"), encoding="utf-8") as f:
        return f.read()


class ReActAgentX(ReActAgent):
    """`ReActAgent`，但模型执行重定向到 01 的 `ModelInterface.generate()`。"""

    def __init__(self, name: str, inner: ModelInterface, sys_prompt: str, toolkit) -> None:
        super().__init__(
            name=name,
            model_config_name=PLACEHOLDER_CONFIG_NAME,
            service_toolkit=toolkit,
            sys_prompt=sys_prompt,
            verbose=False,
        )
        # 覆盖 AgentBase 加载的占位模型，改走 01 统一入口。
        self.model = AgentarModel(inner)


class DialogAgentX(DialogAgent):
    """`DialogAgent`，但模型执行重定向到 01 的 `ModelInterface.generate()`。"""

    def __init__(self, name: str, inner: ModelInterface, sys_prompt: str) -> None:
        super().__init__(
            name=name,
            model_config_name=PLACEHOLDER_CONFIG_NAME,
            sys_prompt=sys_prompt,
            verbose=False,
        )
        self.model = AgentarModel(inner)


def build_agents(inner: ModelInterface) -> dict:
    """构建全部 9 个智能体，统一注入 01 的模型对象（`inner`）。

    Args:
        inner (`ModelInterface`): 由 01 的 `get_model()` 返回的统一模型对象。

    Returns:
        `dict`: key 为角色名，value 为对应 agent 实例。
    """
    agents: dict = {}

    # —— 协调者（DialogAgent，路由 / 意图识别） ——
    agents["coordinator"] = DialogAgentX(
        name="Coordinator",
        inner=inner,
        sys_prompt=_load_prompt("coordinator"),
    )

    # —— 5 个领域专家（ReActAgent） ——
    for scene, prompt_file in [
        ("Banking", "banking"),
        ("Securities", "securities"),
        ("Insurance", "insurance"),
        ("Trust", "trust"),
        ("MutualFunds", "mutualfunds"),
    ]:
        agents[scene] = ReActAgentX(
            name=f"{scene}Expert",
            inner=inner,
            sys_prompt=_load_prompt(prompt_file),
            toolkit=build_toolkit(scene),
        )

    # —— 3 个支撑角色（ReActAgent） ——
    agents["rag"] = ReActAgentX(
        name="RAGRetriever",
        inner=inner,
        sys_prompt=_load_prompt("rag"),
        toolkit=build_toolkit("rag"),
    )
    agents["review"] = ReActAgentX(
        name="ComplianceReviewer",
        inner=inner,
        sys_prompt=_load_prompt("review"),
        toolkit=build_toolkit("review"),
    )
    agents["risk"] = ReActAgentX(
        name="RiskController",
        inner=inner,
        sys_prompt=_load_prompt("risk"),
        toolkit=build_toolkit("risk"),
    )

    return agents


# 合法场景清单（供路由解析使用）。
SCENES = ["Banking", "Securities", "Insurance", "Trust", "MutualFunds"]
