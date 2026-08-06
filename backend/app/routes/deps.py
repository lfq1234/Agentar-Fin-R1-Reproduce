"""09-用户系统与数据隔离：当前用户依赖（对齐 full-stack-fastapi-template 的 api/deps.py）。

- ``SessionDep`` 复用本项目的 ``app.db.models.get_db``（异步 ``AsyncSession``）。
- ``get_current_user`` 解码 Bearer 令牌 → 查库 → 返回 ``User``；任何失败返回 403/404/503。
- 禁用用户返回 403（评审 M3 统一口径）。
- ``CurrentUser`` 直接作为 ``Depends`` 注入受保护路由。
"""
from __future__ import annotations

from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security
from app.db.models import User, get_db

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login/access-token",
    auto_error=True,
)

SessionDep = Depends(get_db)
TokenDep = Depends(reusable_oauth2)


async def get_current_user(
    session: Optional[AsyncSession] = SessionDep,
    token: str = TokenDep,
) -> User:
    if session is None:
        # 鉴权依赖数据库；db.enabled=false 时不提供鉴权（需求：无库则无用户体系）。
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="数据库未启用，鉴权不可用")
    try:
        payload = jwt.decode(token, security.get_secret_key(), algorithms=[security.ALGORITHM])
        user_id = int(payload.get("sub"))
    except (jwt.InvalidTokenError, TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="凭证无效或已过期")
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已停用")
    return user


CurrentUser = Depends(get_current_user)
