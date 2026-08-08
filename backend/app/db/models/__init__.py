"""聚合 DB 模型与连接，供外部 import。

表模型（table=True）与 API 校验模型（非表 SQLModel）同处本包：
SQLModel 底层即 Pydantic，非表模型天然可作为请求/响应校验层，无需独立 `app/schemas`。
Base 继承范式：ConversationBase / MessageBase 承载共享字段，表模型与契约均继承之。
"""
from app.db.models.connection import async_session_maker, engine, get_db, init_db
from app.db.models.conversation import (
    ChatRequest,
    ChatResponse,
    Conversation,
    ConversationBase,
)
from app.db.models.message import (
    AnalyzeRequest,
    AnalyzeResponse,
    MessageBase,
)
from app.db.models.user import User, UserCreate, UserPublic, UserRegister

__all__ = [
    "engine",
    "async_session_maker",
    "get_db",
    "init_db",
    "User",
    "UserCreate",
    "UserPublic",
    "UserRegister",
    "Conversation",
    "ConversationBase",
    "MessageBase",
    "ChatRequest",
    "ChatResponse",
    "AnalyzeRequest",
    "AnalyzeResponse",
]
