"""02-多智能体基础框架：对外导出。

真实框架入口为 `run()`（异步，返回 `AgentResult`），由后续「服务层接入」需求包装为
HTTP 接口；`routes/chat.py` 当前仍调用 `app.agent.runner.run_agent`（stub 兼容壳），
不在本期修改范围。
"""
from app.agent.model_bridge import AgentarModel
from app.agent.retrieval import Passage
from app.agent.system import AgentResult, get_system, run

__all__ = ["run", "get_system", "AgentResult", "AgentarModel", "Passage"]
