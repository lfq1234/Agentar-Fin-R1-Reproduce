"""User 表模型与鉴权契约（SQLModel）。

03 仅用 ``username`` 标识对话归属；09 补齐身份字段：
- ``hashed_password``：pwdlib(Argon2/Bcrypt) 哈希，空字符串表示遗留假用户（不可登录）。
- ``is_active``：禁用后登录/鉴权返回 403。
- ``UserCreate`` / ``UserRegister`` / ``UserPublic``：请求与响应契约（不入库）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, nullable=False, index=True)
    email: Optional[str] = Field(default=None, nullable=True)
    hashed_password: str = Field(default="", nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=_utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=_utcnow, nullable=False)


class UserCreate(SQLModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    email: Optional[str] = Field(default=None, max_length=255)


class UserRegister(UserCreate):
    pass


class UserPublic(SQLModel):
    id: int
    username: str
    email: Optional[str] = None
