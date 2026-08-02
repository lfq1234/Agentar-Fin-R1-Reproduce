"""Message 表模型与 analyze 契约（SQLModel 兼作校验）。

采用 SQLModel 的 Base 继承范式：
- MessageBase 定义共享字段（scene）；
- Message（表模型）、AnalyzeRequest（入参）、AnalyzeResponse（出参）都继承它。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlmodel import Field, ForeignKey, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MessageBase(SQLModel):
    """消息共享字段：表模型与 analyze 契约共用 scene。"""

    scene: Optional[str] = Field(default=None, nullable=True)


class Message(MessageBase, table=True):
    __tablename__ = "messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(
        foreign_key="conversations.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )
    role: str = Field(default="user", nullable=False)  # user / assistant / system
    content: str = Field(default="", nullable=False)
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)


class AnalyzeRequest(MessageBase):
    """/v1/analyze 请求契约（非表模型，兼作 FastAPI 请求体校验）。"""

    message: str = Field(min_length=1)


class AnalyzeResponse(MessageBase):
    """/v1/analyze 响应契约（非表模型，兼作响应序列化）。

    scene 回显被分析请求的场景。
    """

    intent: Optional[str] = None
    slots: Dict[str, str] = Field(default_factory=dict)
    tool_plan: List[str] = Field(default_factory=list)
    expression: str = ""
