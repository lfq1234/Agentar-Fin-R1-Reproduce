"""HTTP routes for chat / financial task analysis."""
from __future__ import annotations

from fastapi import APIRouter

from app.agent.runner import run_agent
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, ChatRequest, ChatResponse

router = APIRouter(prefix="/v1", tags=["agent"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    reply = run_agent(req.message, scene=req.scene, task=req.task)
    return ChatResponse(reply=reply)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Run the Agent capability pipeline: intent -> slot -> tool-plan -> expression."""
    result = run_agent(req.message, scene=req.scene, task="ConsultationQA", structured=True)
    return AnalyzeResponse(**result)
