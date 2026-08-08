"""Message 契约（analyze 用，非表模型）。

消息表（messages）已并入 ``conversations.data``（单表记录聊天，见 03/07 收口方案）：
聊天正文 + 多专家执行轨迹全部自包含于 ``conversations.data``(JSON) 字段，
本文件仅保留 analyze 接口的请求/响应契约，不再定义 Message 表模型。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from sqlmodel import Field, SQLModel


class MessageBase(SQLModel):
    """消息共享字段：analyze 契约共用 scene。"""

    scene: Optional[str] = Field(default=None, nullable=True)


class AnalyzeRequest(MessageBase):
    """/api/v1/analyze 请求契约（非表模型，兼作 FastAPI 请求体校验）。"""

    message: str = Field(min_length=1)


class AnalyzeResponse(MessageBase):
    """/api/v1/analyze 响应契约（非表模型，兼作响应序列化）。

    scene 回显被分析请求的场景。
    """

    intent: Optional[str] = None
    slots: Dict[str, str] = Field(default_factory=dict)
    tool_plan: List[str] = Field(default_factory=list)
    expression: str = ""
