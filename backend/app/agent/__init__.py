"""02-多智能体基础框架：对外导出。

真实框架入口为 `run()`（异步，返回 `AgentResult`），由 `services/chat_service.py`
包装为 HTTP 接口（`routes/chat.py` 经 `chat_service` 调用）。专家互相通讯编排见
`app.agent.board.ExpertBoard`（档A 互询 / 档B 圆桌会商），由 `run()` 统一编排。
"""
from app.agent.board import ExpertBoard
from app.agent.model_bridge import AgentarModel
from app.agent.retrieval import Passage
from app.agent.system import AgentResult, get_system, run

__all__ = ["run", "get_system", "AgentResult", "AgentarModel", "Passage", "ExpertBoard"]
