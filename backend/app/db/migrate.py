"""09-用户系统与数据隔离：兼容旧库的轻量迁移。

``init_db()`` 用 ``SQLModel.metadata.create_all``，对已存在的 ``users`` 表不会 ALTER。
本项目保留 ``agentar.db``（用户要求不删除旧数据），故用幂等 ALTER 补齐 09 新增列：
- ``hashed_password TEXT NOT NULL DEFAULT ''``
- ``is_active INTEGER NOT NULL DEFAULT 1``

列已存在时 SQLite 报「duplicate column」，捕获后忽略，保持幂等；不重建、不丢数据。
"""
from __future__ import annotations

from sqlalchemy import text

from app.db.models.connection import engine


async def migrate_user_columns() -> None:
    if engine is None:
        return
    ddls = [
        "ALTER TABLE users ADD COLUMN hashed_password TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1",
    ]
    async with engine.begin() as conn:
        for ddl in ddls:
            try:
                await conn.execute(text(ddl))
            except Exception:
                # 列已存在（重复 ADD COLUMN）→ 忽略，幂等。
                pass
