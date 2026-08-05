"""07-导出与复盘：export_session（md/json）+ get_session_for_review（raf 上下文）。

07 与 raf 职责边界：07 只「记录 + 组织成结构化上下文」，「总结成复盘」是 raf 的 LLM 职责
（技术文档 §6.3，类比 06 把 Passage 喂 raf）。
"""
from __future__ import annotations

import json
from typing import Any

from app.history.models import SessionDetail


def _iter_events(node: dict, depth: int = 0):
    """深度优先展开事件树为扁平列表（带缩进层级）。"""
    yield node, depth
    for child in node.get("children", []):
        yield from _iter_events(child, depth + 1)


def _iter_event_nodes(nodes: list) -> Any:
    """遍历事件列表（每个事件本身即树根），深度优先展开。"""
    for n in nodes or []:
        yield from _iter_events(n, 0)


def render_export(detail: SessionDetail, *, fmt: str = "md") -> str:
    """导出复盘文档（md / json）。"""
    if fmt == "json":
        return json.dumps(_detail_to_dict(detail), ensure_ascii=False, indent=2)

    lines: list[str] = []
    meta = detail.meta
    lines.append("# 会话复盘文档")
    lines.append("")
    if meta:
        lines.append(f"- 会话 ID：`{detail.conversation_id}`")
        lines.append(f"- 场景：{meta.scene or '—'}")
        lines.append(f"- 标题：{meta.title or '—'}")
        lines.append(f"- 状态：{meta.status}")
        lines.append(f"- 消息数：{meta.msg_count}　累计 token：{meta.total_tokens}")
        lines.append("")
    # 主消息流
    lines.append("## 主消息流")
    lines.append("")
    for m in detail.messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        lines.append(f"**{role}**：{content}")
    lines.append("")
    # trace
    if detail.has_trace and detail.trace:
        lines.append("## 执行轨迹（trace）")
        lines.append("")
        for run in detail.trace:
            lines.append(
                f"### run `{run['run_id'][:8]}` · model={run.get('model')} · "
                f"耗时 {run.get('duration_ms')}ms · token {run.get('total_tokens')} · "
                f"状态 {run.get('status')}"
            )
            for node, depth in _iter_event_nodes(run["events"]):
                if node.get("children") is None:
                    continue
                pad = "  " * depth
                head = f"{pad}- [{node.get('type')}] {node.get('agent') or ''}"
                lines.append(head)
                if node.get("summary_in"):
                    lines.append(f"{pad}  - 输入：{node['summary_in']}")
                if node.get("summary_out"):
                    lines.append(f"{pad}  - 输出：{node['summary_out']}")
                meta_d = node.get("meta")
                if meta_d:
                    lines.append(f"{pad}  - 结构化：{json.dumps(meta_d, ensure_ascii=False)}")
    lines.append("")
    return "\n".join(lines)


def render_review_context(detail: SessionDetail) -> str:
    """生成供 raf 总结智能体消费的结构化复盘上下文（带溯源）。"""
    parts: list[str] = []
    parts.append("【会话复盘上下文】")
    meta = detail.meta
    if meta:
        parts.append(f"场景：{meta.scene or '未知'}；标题：{meta.title or '未知'}")
    parts.append("\n=== 用户诉求 ===")
    for m in detail.messages:
        if m.get("role") == "user":
            parts.append(f"- {m.get('content', '')}")
    parts.append("\n=== 各智能体结论 / RAG 依据 / 审核风控 ===")
    if detail.has_trace and detail.trace:
        for run in detail.trace:
            for node, _ in _iter_event_nodes(run["events"]):
                if node.get("children") is None:
                    continue
                t = node.get("type")
                agent = node.get("agent") or ""
                if t in ("user", "assistant"):
                    continue
                summary = node.get("summary_out") or node.get("summary_in") or ""
                meta_d = node.get("meta") or {}
                rag = ""
                if t == "rag_hit" and meta_d:
                    rag = f"（溯源：{meta_d.get('doc_id')} / {meta_d.get('title')} / score={meta_d.get('score')}）"
                parts.append(f"- [{t}/{agent}] {summary} {rag}".strip())
    return "\n".join(parts)


def _detail_to_dict(detail: SessionDetail) -> dict:
    out: dict[str, Any] = {
        "conversation_id": detail.conversation_id,
        "meta": (
            {
                "scene": detail.meta.scene,
                "title": detail.meta.title,
                "status": detail.meta.status,
                "msg_count": detail.meta.msg_count,
                "total_tokens": detail.meta.total_tokens,
                "first_at": detail.meta.first_at,
                "last_at": detail.meta.last_at,
            }
            if detail.meta
            else None
        ),
        "messages": detail.messages,
        "trace": detail.trace,
        "has_trace": detail.has_trace,
    }
    return out
