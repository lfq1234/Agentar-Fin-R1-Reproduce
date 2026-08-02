"""03-数据层：异步引擎 / 会话 / get_db 依赖 / init_db，含 enabled 门控。

- db.enabled=false：引擎/会话不构建，get_db 产出 None，系统无库也能启动并响应。
- db.enabled=true ：按 config.db 构建异步 SQLite 引擎（sqlite+aiosqlite），提供会话依赖。
- 开发/测试期 init_db() 用 SQLModel.metadata.create_all 建表；生产以 schema.sql 为准。
- 异步参考 tiangolo/full-stack-fastapi-template 的 core/db.py，但保留本项目的 enabled 短路设计。
"""
from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator, Optional

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import config

_db_cfg: dict = config.get("db", {}) or {}
_ENABLED: bool = bool(_db_cfg.get("enabled", False))

engine: Optional[AsyncEngine] = None
async_session_maker: Optional[async_sessionmaker] = None

if _ENABLED:
    # 仅支持 SQLite 后端：sqlite+aiosqlite://（异步驱动）
    _path = Path(str(_db_cfg.get("sqlite_path", "./agentar.db"))).resolve()
    _url = f"sqlite+aiosqlite:///{_path}"
    _echo = bool(_db_cfg.get("echo", False))
    engine = create_async_engine(_url, echo=_echo, future=True)

    # SQLite 默认不强制外键，需开启 PRAGMA 才能级联删除（监听底层 sync_engine）。
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_con, _record) -> None:  # noqa: ANN001
        dbapi_con.execute("PRAGMA foreign_keys=ON")

    async_session_maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )


async def init_db() -> None:
    """db.enabled=true 时按 SQLModel 元数据建表（开发/测试期）。

    生产环境以 app/db/schema.sql 的可执行 DDL 为准，部署时执行该脚本建表。
    """
    if not _ENABLED or engine is None:
        return
    from sqlmodel import SQLModel

    # 触发模型模块导入，确保表注册到 SQLModel.metadata。
    from app.db.models import conversation, message, user  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_db() -> AsyncGenerator[Optional[AsyncSession], None]:
    """FastAPI 依赖：db.enabled=false 时产出 None（服务层据此短路落库）。"""
    if not _ENABLED or async_session_maker is None:
        yield None
        return
    async with async_session_maker() as session:
        yield session
