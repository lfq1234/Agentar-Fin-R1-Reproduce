"""09-用户系统与数据隔离：User CRUD（镜像 full-stack-fastapi-template 的 crud.user）。

- ``authenticate`` 统一返回 ``User | None``，不区分「账号不存在 / 密码错」（需求 FR9，防用户枚举）。
- 创建前由路由层校验唯一性（``get_by_username``），本层只负责落库。
"""
from __future__ import annotations

from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, UserCreate


async def get_by_username(session: AsyncSession, username: str) -> User | None:
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def authenticate(session: AsyncSession, username: str, password: str) -> User | None:
    user = await get_by_username(session, username)
    if user is None:
        return None
    from app.core import security

    if not security.verify_password(password, user.hashed_password):
        return None
    return user


async def create_user(session: AsyncSession, user_in: UserCreate) -> User:
    from app.core import security

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=security.get_password_hash(user_in.password),
        is_active=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user
