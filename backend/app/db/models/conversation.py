"""Conversation 表模型与 chat 契约（SQLModel 兼作校验）。

采用 SQLModel 的 Base 继承范式（参考 tiangolo/full-stack-fastapi-template）：
- ConversationBase 定义共享字段（scene）；
- Conversation（表模型）、ChatRequest（入参）、ChatResponse（出参）都继承它，字段只写一次。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlmodel import Field, ForeignKey, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationBase(SQLModel):
    """会话共享字段：表模型与请求/响应契约都继承。"""

    scene: Optional[str] = Field(default=None, nullable=True)  # 会话级场景，首创为准


class Conversation(ConversationBase, table=True):
    __tablename__ = "conversations"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(
        foreign_key="users.id",
        nullable=False,
        index=True,
        ondelete="CASCADE",
    )
    title: Optional[str] = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow, nullable=False)


class ChatRequest(ConversationBase):
    """/api/v1/chat 请求契约（非表模型，兼作 FastAPI 请求体校验）。"""

    message: str = Field(min_length=1)
    user_id: Optional[int] = None  # 临时方案：请求体携带；接鉴权后改 token 解析
    conversation_id: Optional[int] = None  # 续聊


class ChatResponse(ConversationBase):
    """/api/v1/chat 响应契约（非表模型，兼作响应序列化）。

    scene 回显会话级场景，便于前端对齐上下文。
    """

    conversation_id: Optional[int] = None
    reply: str
    compliance_notes: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
