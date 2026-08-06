"""DB 启用态下的落库验证（评审门禁问题5：未知 user_id 自动 upsert + 消息落库）。

异步版本：用内存 SQLite（aiosqlite + StaticPool 保证单库）自建异步引擎与会话，
直接调用 chat_service（其落库使用 AsyncSession）。
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, select

from app.agent.system import AgentResult
from app.db.models import ChatRequest, Conversation, Message, User
from app.services import chat_service  # 需要模块对象本身（monkeypatch 其内的 agent_run）


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


async def _fake_run(message, scene=None, structured=False, user_id=None, use_personal_docs=False) -> AgentResult:
    return AgentResult(reply="这是回复", compliance_notes=["c"], risk_flags=["r"])


@pytest.mark.asyncio
async def test_persist_new_user_and_messages(
    monkeypatch: pytest.MonkeyPatch, session: AsyncSession
) -> None:
    monkeypatch.setattr(chat_service, "agent_run", _fake_run)

    req = ChatRequest(message="你好", user_id=42, scene="Banking")
    resp = await chat_service.chat(req, session)

    assert resp.conversation_id is not None
    # 未知 user_id 自动 upsert
    assert await session.get(User, 42) is not None
    # 会话创建，scene 以首次请求为准
    conv = await session.get(Conversation, resp.conversation_id)
    assert conv is not None
    assert conv.scene == "Banking"
    # 两条消息落库（user + assistant）
    rows = (await session.execute(select(Message).where(Message.conversation_id == conv.id))).scalars().all()
    assert len(rows) == 2
    assert {m.role for m in rows} == {"user", "assistant"}
    # 消息级 scene 继承会话 scene
    assert all(m.scene == "Banking" for m in rows)


@pytest.mark.asyncio
async def test_skip_persist_when_db_none(
    monkeypatch: pytest.MonkeyPatch, session: AsyncSession
) -> None:
    monkeypatch.setattr(chat_service, "agent_run", _fake_run)

    # db=None（enabled=false 短路）：不落库，仍返回 reply
    req = ChatRequest(message="你好", user_id=7)
    resp = await chat_service.chat(req, None)
    assert resp.conversation_id is None
    assert resp.reply == "这是回复"
    assert (await session.execute(select(User))).scalars().all() == []


@pytest.mark.asyncio
async def test_message_scene_override(
    monkeypatch: pytest.MonkeyPatch, session: AsyncSession
) -> None:
    monkeypatch.setattr(chat_service, "agent_run", _fake_run)

    # 第一轮：建立 Banking 会话
    r1 = await chat_service.chat(
        ChatRequest(message="你好", user_id=99, scene="Banking"), session
    )
    conv = await session.get(Conversation, r1.conversation_id)
    assert conv.scene == "Banking"
    # 第二轮：续聊同一会话，显式传 Securities → 该轮两条消息覆盖为 Securities，会话 scene 不变
    r2 = await chat_service.chat(
        ChatRequest(
            message="续问", user_id=99, scene="Securities", conversation_id=conv.id
        ),
        session,
    )
    assert r2.conversation_id == conv.id
    assert (await session.get(Conversation, conv.id)).scene == "Banking"  # 会话级不变

    rows = (await session.execute(select(Message).where(Message.conversation_id == conv.id))).scalars().all()
    assert len(rows) == 4  # 两轮各 2 条
    by_scene: dict = {}
    for m in rows:
        by_scene.setdefault(m.scene, set()).add(m.role)
    assert by_scene["Banking"] == {"user", "assistant"}
    assert by_scene["Securities"] == {"user", "assistant"}
