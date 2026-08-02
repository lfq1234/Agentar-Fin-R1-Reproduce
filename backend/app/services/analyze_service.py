"""analyze 服务：消费 02 框架，返回结构化四字段（不落库）。"""
from __future__ import annotations

from app.agent import run as agent_run
from app.db.models import AnalyzeRequest, AnalyzeResponse


async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    result = await agent_run(message=req.message, scene=req.scene, structured=True)
    return AnalyzeResponse(
        intent=result.intent,
        slots=result.slots or {},
        tool_plan=result.tool_plan or [],
        expression=result.expression or "",
    )
