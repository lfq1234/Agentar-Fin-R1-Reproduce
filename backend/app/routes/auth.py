"""09-用户系统与数据隔离：认证路由（register / login / me）。

路径（``main.py`` 以 ``prefix="/api"`` 注册，本路由内前缀 ``/v1/auth``）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | ``/api/v1/auth/register`` | 注册（201，响应不含密码） |
| POST | ``/api/v1/auth/login/access-token`` | OAuth2 密码流登录，返回 Bearer 令牌 |
| GET  | ``/api/v1/auth/me`` | 当前用户（需令牌） |
"""
from __future__ import annotations

from typing import Annotated

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.core import security
from app.crud import user as crud
from app.db.models import User, UserCreate, UserPublic
from app.routes.deps import CurrentUser, SessionDep

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/register", response_model=UserPublic, status_code=201)
async def register(user_in: UserCreate, session: AsyncSession = SessionDep) -> UserPublic:
    if await crud.get_by_username(session, user_in.username):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="账号已存在")
    user = await crud.create_user(session, user_in)
    return UserPublic(id=user.id, username=user.username, email=user.email)


@router.post("/login/access-token")
async def login_access_token(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: AsyncSession = SessionDep,
) -> dict:
    user = await crud.authenticate(session, form.username, form.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="账号或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已停用")
    minutes = int((config.get("security", {}) or {}).get("access_token_expire_minutes", 60))
    token = security.create_access_token(user.id, timedelta(minutes=minutes))
    return {"access_token": token, "token_type": "bearer"}


@router.get("/me", response_model=UserPublic)
async def read_me(current_user: User = CurrentUser) -> UserPublic:
    return UserPublic(id=current_user.id, username=current_user.username, email=current_user.email)
