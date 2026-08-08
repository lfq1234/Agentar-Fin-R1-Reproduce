"""02-多智能体基础框架：专家协作板（专家互相通讯编排）。

基于 AgentScope 0.1.6 的「Agent 可互调」原语（`AgentBase.__call__` 为 `async`），
在编排层（``run()``）实现专家之间的通讯，无需从零造轮子：

- 档A 专家互询：``consult(peer_name, msg)`` 直接 ``await`` 同级专家，取其文本意见；
- 档B 圆桌会商：``roundtable(peer_names, msg)`` 顺序调用多位专家收集意见。

所有调用发生在 ``run()`` 的 ``with scoped(...)`` 块内，``ContextVar`` 随调用栈透传，
子专家内部的 ``lookup_knowledge`` 自动落在与主答**相同**的 RAG 作用域
（FR-EX4，天然越权防护，无需额外传参）。

> 设计取舍：专家互询放在编排层（``run()``）而非注册成 ReAct 工具，原因——
> ``ReActAgent`` 的工具调用是**同步**执行（``func.processed_func(**kwargs)`` 不 await），
> 若把互询做成工具，需在同步函数里跑异步 agent 调用（线程 + 上下文复制），偏重。
> 编排层经 ``ExpertBoard`` 协调，全程 async、ContextVar 天然透传，改动最小。
"""
from __future__ import annotations

import asyncio

from agentscope.message import Msg

from app.agent.model_bridge import extract_content

# FR-EX5：会商 / 互询参与专家数上限，避免弱模型空转或成本 / 延迟失控。
MAX_CONSULT_ROUNDS = 3

# 进程级串行锁：本地模型（LocalTransformerModel.generate 为同步阻塞调用）+ 共享单例
# agent（含对话记忆）不允许并发推理。``system._call_agent`` 与本会商板的 ``_call_agent``
# 共用此锁，避免两条编排路径并发调用同一批单例 agent 导致对话记忆串味 / 单 GPU 推理打架。
_MODEL_LOCK = asyncio.Lock()


async def _call_agent(agent, msg: Msg) -> str:
    """调用 agent 并提取文本（与 ``system._call_agent`` 同语义、共用 ``_MODEL_LOCK``）。

    AgentScope 0.1.6 的 ``agent.__call__`` 是同步函数（``@async_func`` 仅为 RPC 标记，
    不改变同步语义），其内部的 ``model.generate()`` 会阻塞调用线程。这里用
    ``asyncio.to_thread`` 把整次 agent 应答搬离 uvicorn 事件循环，使服务期间事件循环
    不被占住——``/health``、登录、第二个 ``/chat`` 等请求仍可正常响应；并以 ``_MODEL_LOCK``
    串行化，避免并发修改共享单例 agent 的记忆（不改 01 模型路径）。
    """
    async with _MODEL_LOCK:
        out = await asyncio.to_thread(agent, msg)
    if asyncio.iscoroutine(out):
        out = await out
    return extract_content(out)


class ExpertBoard:
    """持有全部领域专家引用，提供互询 / 会商编排。

    合成复用 coordinator 角色（不新增 agent），以稳定 02 框架的 9 角色契约。
    """

    def __init__(self, agents: dict) -> None:
        # agents：与 system._AgentSystem.agents 同结构（含领域专家与 coordinator）。
        self.agents = agents

    async def consult(self, peer_name: str, msg: Msg) -> str:
        """档A 专家互询：直接 await 同级专家，返回其文本意见。

        Args:
            peer_name (`str`): 被咨询专家的角色 key（如 ``"Insurance"``）。
            msg (`Msg`): 转发给该专家的提问。

        Returns:
            `str`: 该专家返回的文本意见。
        """
        return await _call_agent(self.agents[peer_name], msg)

    async def roundtable(self, peer_names: list[str], msg: Msg) -> list[tuple[str, str]]:
        """档B 圆桌会商：顺序调用多位专家，返回 ``[(专家名, 意见)]``。

        Args:
            peer_names (`list[str]`): 参与会商的领域专家角色 key 列表。
            msg (`Msg`): 共享的提问（含上下文）。

        Returns:
            `list[tuple[str, str]]`: 各专家的名称与文本意见；受 ``MAX_CONSULT_ROUNDS`` 限制。
        """
        opinions: list[tuple[str, str]] = []
        for name in peer_names[:MAX_CONSULT_ROUNDS]:
            peer = self.agents[name]
            opinions.append((peer.name, await _call_agent(peer, msg)))
        return opinions
