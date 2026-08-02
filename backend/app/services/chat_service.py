"""chat 服务：消费 02 框架 + 落库（User upsert）。

- 调用 02 的 `async app.agent.run()`（必须 await）。
- db 可用且提供 user_id 时：upsert User → 取得/创建 Conversation → 写 user/assistant 两条 Message。
- db 不可用或 user_id 为空时：跳过落库，仍返回 reply（conversation_id 为 None）。
- 落库使用异步 AsyncSession（await 提交 / 刷新）。
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import run as agent_run
from app.db.models import ChatRequest, ChatResponse, Conversation, Message, User


async def _persist(db: AsyncSession, req: ChatRequest, reply: str) -> int:
    """落库并返回 conversation_id（未知 user_id 自动 upsert）。"""
    # (a) upsert User
    user = await db.get(User, req.user_id)
    if user is None:
        user = User(id=req.user_id, username=f"user_{req.user_id}")
        db.add(user)
        await db.flush()

    # (b) 取得或创建会话
    conv = None
    if req.conversation_id is not None:
        conv = await db.get(Conversation, req.conversation_id)
    if conv is None:
        conv = Conversation(user_id=user.id, scene=req.scene)
        db.add(conv)
        await db.flush()

    # (c) 写入两条消息；消息级 scene 默认继承会话 scene，仅显式传参时覆盖
    msg_scene = req.scene if req.scene is not None else conv.scene
    db.add(
        Message(conversation_id=conv.id, role="user", content=req.message, scene=msg_scene)
    )
    db.add(
        Message(conversation_id=conv.id, role="assistant", content=reply, scene=msg_scene)
    )

    await db.commit()
    await db.refresh(conv)
    return conv.id


async def chat(req: ChatRequest, db: Optional[AsyncSession]) -> ChatResponse:
    result = await agent_run(message=req.message, scene=req.scene, structured=False)
    reply = result.reply
    compliance_notes = result.compliance_notes or []
    risk_flags = result.risk_flags or []

    conversation_id: Optional[int] = None
    if db is not None and req.user_id is not None:
        conversation_id = await _persist(db, req, reply)

    return ChatResponse(
        reply=reply,
        conversation_id=conversation_id,
        compliance_notes=compliance_notes,
        risk_flags=risk_flags,
    )
