"""chat 服务：消费 02 框架 + 落库（User upsert）。

- 调用 02 的 `async app.agent.run()`（必须 await）。
- db 可用且提供 user_id 时：upsert User → 取得/创建 Conversation →
  把 user/assistant 两条消息追加进 ``conversations.data``(JSON) 单字段。
- 多专家执行轨迹（trace）由 07 history hooks 在 ``record_run`` 中挂到该轮助手消息上，
  二者同落 ``conversations.data``，实现「一次会话一条记录」（评审：避免会话记录过度拆分）。
- db 不可用或 user_id 为空时：跳过落库，仍返回 reply（conversation_id 为 None）。
- 落库使用异步 AsyncSession（await 提交 / 刷新）。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import run as agent_run
from app.db.models import ChatRequest, ChatResponse, Conversation, User


def _dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# 多人对话模式：给各智能体步骤分配头像与中文名，前端按 agent 字段渲染。
_AGENT_META: dict[str, tuple[str, str]] = {
    "Coordinator": ("🤖", "协调者"),
    "Direct": ("🤖", "Agentar"),
    "rag": ("🔍", "资料检索"),
    "RAGRetriever": ("🔍", "资料检索"),
    "Banking": ("🏦", "银行专家"),
    "BankingExpert": ("🏦", "银行专家"),
    "Securities": ("📈", "证券专家"),
    "SecuritiesExpert": ("📈", "证券专家"),
    "Insurance": ("🛡️", "保险专家"),
    "InsuranceExpert": ("🛡️", "保险专家"),
    "Trust": ("🏛️", "信托专家"),
    "TrustExpert": ("🏛️", "信托专家"),
    "MutualFunds": ("🧺", "基金专家"),
    "MutualFundsExpert": ("🧺", "基金专家"),
    "review": ("✅", "合规审核"),
    "ComplianceReviewer": ("✅", "合规审核"),
    "risk": ("⚠️", "风控"),
    "RiskController": ("⚠️", "风控"),
    "user": ("👤", "用户"),
}


def _agent_meta(agent_key: str) -> dict[str, str]:
    avatar, name = _AGENT_META.get(agent_key, ("🤖", agent_key))
    return {"avatar": avatar, "name": name}


def _mention_name(agent_key: str | None) -> str | None:
    if agent_key is None or agent_key == "用户":
        return "用户"
    return _agent_meta(agent_key)["name"]


def _build_messages(result: Any) -> list[dict[str, Any]]:
    """把 AgentResult 的 trace + 审核/风控输出拆成前端可渲染的多人对话消息列表。

    每个非最终步骤的气泡末尾会 @ 下一个智能体；最后一条为整理后的正式助手回复。
    """
    trace: list[dict[str, Any]] = getattr(result, "agent_trace", None) or []
    messages: list[dict[str, Any]] = []

    # 1) 按 trace 顺序生成智能体气泡（跳过内部步骤：revise 最终改写/route 路由判定/rag 检索，
    #    避免把"场景判定：Direct"等编排内部信息当作对话气泡暴露给用户；只保留专家意见、
    #    合规审核、风控与最终回复，符合"不该每个都调、直接给答案"的诉求）。
    non_final_steps = [s for s in trace if s.get("type") not in ("revise", "route", "rag")]
    for i, step in enumerate(non_final_steps):
        agent_key = step.get("agent") or "Coordinator"
        meta = _agent_meta(agent_key)
        next_key = non_final_steps[i + 1].get("agent") if i + 1 < len(non_final_steps) else None
        # 如果后面还有专家步骤，则 @ 下一个专家；否则指向合规/风控/最终环节
        if next_key is None:
            next_key = "review" if getattr(result, "compliance_notes", None) else (
                "risk" if getattr(result, "risk_flags", None) else "Coordinator"
            )
        next_name = _mention_name(next_key)
        content = step.get("content", "")
        messages.append({
            "role": "agent",
            # AgentScope 名字键（如 BankingExpert），前端按 AGENT_DISPLAY 映射中文名+头像。
            "agent": agent_key,
            "avatar": meta["avatar"],
            "type": step.get("type", ""),
            "content": f"{content}\n\n@{next_name}".strip(),
            "mention": next_name,
        })

    # 2) 合规审核与风控作为独立气泡插入（建议式，不阻断）
    for note in getattr(result, "compliance_notes", None) or []:
        if note:
            meta = _agent_meta("review")
            messages.append({
                "role": "agent",
                # 用 AgentScope 名字键 ComplianceReviewer（与前端 AGENT_DISPLAY 对齐）
                "agent": "ComplianceReviewer",
                "avatar": meta["avatar"],
                "type": "review",
                "content": f"{note}\n\n@{_mention_name('risk')}".strip(),
                "mention": _mention_name("risk"),
            })
    for flag in getattr(result, "risk_flags", None) or []:
        if flag:
            meta = _agent_meta("risk")
            messages.append({
                "role": "agent",
                "agent": "RiskController",
                "avatar": meta["avatar"],
                "type": "risk",
                "content": f"{flag}\n\n@{_mention_name('Coordinator')}".strip(),
                "mention": _mention_name("Coordinator"),
            })

    # 3) 最终正式回复：由 revise 步骤的作答角色或协调者输出，不再 @ 任何人。
    #    trace 为空意味着走 Direct 直答通道，使用 Direct/Agentar 身份。
    final_agent_key = "Direct" if not trace else "Coordinator"
    for step in reversed(trace):
        if step.get("type") == "revise":
            final_agent_key = step.get("agent") or "Coordinator"
            break
    final_meta = _agent_meta(final_agent_key)
    messages.append({
        "role": "assistant",
        # AgentScope 名字键（Direct / Coordinator），前端映射出"谁综合作答"。
        "agent": final_agent_key,
        "avatar": final_meta["avatar"],
        "type": "final",
        "content": getattr(result, "reply", ""),
        "mention": None,
    })
    return messages


async def _persist(
    db: AsyncSession,
    req: ChatRequest,
    reply: str,
    user_id: int,
    agent_messages: Optional[list[dict[str, Any]]] = None,
) -> int:
    """落库并返回 conversation_id；user_id 来自鉴权后的当前用户（09）。

    聊天记录（用户/助手消息）写入 ``conversations.data`` JSON 单字段，
    不再拆 ``messages`` 表（07 收口方案）。
    """
    # (a) 取已存在的用户（鉴权保证存在；兜底：极端无库路径下重建占位用户，保持兼容）。
    user = await db.get(User, user_id)
    if user is None:
        user = User(id=user_id, username=f"user_{user_id}", is_active=True)
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

    # (c) 把 user + 多人对话 agent 消息 + 最终 assistant 消息追加进 conversations.data
    msg_scene = req.scene if req.scene is not None else conv.scene
    now = datetime.now(timezone.utc)
    ms = _dt_to_ms(now)
    data = json.loads(conv.data) if conv.data else {"messages": []}
    data.setdefault("messages", [])
    data["messages"].append(
        {"role": "user", "content": req.message, "scene": msg_scene, "created_at": ms}
    )
    for idx, am in enumerate(agent_messages or []):
        data["messages"].append(
            {
                "role": am.get("role", "agent"),
                "content": am.get("content", ""),
                "agent": am.get("agent"),
                "avatar": am.get("avatar"),
                "mention": am.get("mention"),
                "type": am.get("type"),
                "scene": msg_scene,
                "created_at": ms + idx + 1,
            }
        )
    data["messages"].append(
        {"role": "assistant", "content": reply, "scene": msg_scene, "created_at": ms + len(agent_messages or []) + 1}
    )
    conv.data = json.dumps(data, ensure_ascii=False)
    conv.updated_at = now
    if not conv.title and req.message:
        conv.title = req.message[:50]

    await db.commit()
    await db.refresh(conv)
    return conv.id


async def chat(req: ChatRequest, db: Optional[AsyncSession], user_id: Optional[int] = None) -> ChatResponse:
    # 09：user_id 优先取鉴权后的当前用户；兼容未接鉴权的旧调用方（回退 req.user_id）。
    uid = user_id if user_id is not None else req.user_id
    # 08：use_personal_docs 仅在同时给出 user_id 时生效（无归属 → 不可检索个人文档）。
    use_personal_docs = bool(getattr(req, "use_personal_docs", False)) and uid is not None
    result = await agent_run(
        message=req.message,
        scene=req.scene,
        structured=False,
        user_id=uid,
        use_personal_docs=use_personal_docs,
    )
    reply = result.reply
    compliance_notes = result.compliance_notes or []
    risk_flags = result.risk_flags or []

    # 02 多人对话：把一次编排拆成有序气泡（含头像与 @next）
    messages = _build_messages(result)

    conversation_id: Optional[int] = None
    if db is not None and uid is not None:
        conversation_id = await _persist(db, req, reply, uid, agent_messages=messages)

    return ChatResponse(
        reply=reply,
        conversation_id=conversation_id,
        compliance_notes=compliance_notes,
        risk_flags=risk_flags,
        agent_trace=result.agent_trace,
        messages=messages,
    )
