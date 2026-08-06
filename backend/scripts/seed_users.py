"""09-用户系统与数据隔离：开发期播种演示账号（可选）。

幂等：账号已存在则跳过。运行（后端目录，需 db.enabled=true）：
    cd backend
    python scripts/seed_users.py

默认播种 alice / bob（密码见 ``DEMO_USERS``）。仅供本地联调，勿用于生产。
"""
from __future__ import annotations

import asyncio

from sqlmodel import select

from app.core import security
from app.db.models import User, get_db

DEMO_USERS = [
    ("alice", "alice1234"),
    ("bob", "bob1234"),
]


async def main() -> None:
    async for session in get_db():
        for username, password in DEMO_USERS:
            existing = await session.execute(select(User).where(User.username == username))
            if existing.scalar_one_or_none() is not None:
                print(f"skip (exists): {username}")
                continue
            session.add(
                User(
                    username=username,
                    hashed_password=security.get_password_hash(password),
                    is_active=True,
                )
            )
            print(f"seeded: {username} / {password}")
        await session.commit()
    print("seed done")


if __name__ == "__main__":
    asyncio.run(main())
