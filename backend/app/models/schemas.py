"""Request / response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    scene: str | None = None  # e.g. Banking / Securities / Insurance / Trust / MutualFunds
    task: str | None = None   # e.g. NER / IntentClassification / SlotFilling / ...


class ChatResponse(BaseModel):
    reply: str


class AnalyzeRequest(BaseModel):
    message: str
    scene: str | None = None


class AnalyzeResponse(BaseModel):
    intent: str | None = None
    slots: dict[str, str] = {}
    tool_plan: list[str] = []
    expression: str = ""
