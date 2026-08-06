"""08 个人文档：知识图谱抽取（LLM ``auto`` / 启发式 ``none`` 两档，永不阻塞入库）。

设计（技术文档 §5.6）：
- ``model="none"``（默认）：纯启发式——文档名作 ``Document`` 根节点，正则抽取金融实体
  （机构 / 产品 / 条款 / 指标 / 人物 / 概念）作节点，根→实体建「提及」边，
  同段共现的异类实体建「相关」边；
- ``model="auto"``：走 01 ``get_model().generate()`` 输出 JSON，解析失败或模型不可达时
  **回退启发式**，不抛错、不阻断 ingest（文档仍 ``status=done``）。

节点 / 边 id 均以 ``doc_id`` 为前缀，保证跨文档唯一且删除文档时可整段清理。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

# 图谱规模上限：避免长文档产生巨图拖垮前端力导向布局
MAX_NODES = 40
MAX_EDGES = 80

NODE_TYPES = ("Document", "Org", "Product", "Clause", "Metric", "Person", "Concept")

# 金融实体启发式规则（中文为主，按类型优先级从具体到宽泛匹配）
_PATTERNS: list[tuple[str, str]] = [
    (
        "Org",
        r"[\u4e00-\u9fa5A-Za-z]{2,12}?(?:银行|证券|保险|基金管理|信托|交易所|证监会|"
        r"银保监会|金融监管总局|人民银行|资产管理|资管|券商)(?:股份有限公司|有限公司)?",
    ),
    (
        "Product",
        r"[\u4e00-\u9fa5A-Za-z0-9]{2,14}?(?:理财产品|信托计划|资管计划|年金险|重疾险|"
        r"寿险|指数基金|货币基金|债券基金|ETF|结构性存款|定期存款|保单)",
    ),
    ("Clause", r"第[一二三四五六七八九十百千零〇\d]{1,6}条"),
    (
        "Metric",
        r"(?:年化收益率|七日年化|净值|风险等级|管理费率|托管费率|申购费|赎回费|"
        r"最大回撤|夏普比率|保额|保费)(?:[:：]?\s*[\dA-Za-z.%万亿]+)?",
    ),
    ("Person", r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|经理|总监|董事长|基金经理)"),
]


@dataclass
class GraphNode:
    id: str
    label: str
    type: str
    source_doc_id: str
    properties: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str
    label: str
    source_doc_id: str


_LLM_PROMPT = """你是金融文档知识图谱抽取器。请从下面的文档片段中抽取实体与关系。

要求：
1. 只输出 JSON，不要任何解释文字或 markdown 代码块标记；
2. 结构为 {{"nodes": [{{"label": "实体名", "type": "类型"}}], "edges": [{{"source": "实体名A", "target": "实体名B", "label": "关系"}}]}}；
3. type 只能取：{types}；
4. 实体名必须原文出现，不得编造；最多 {max_nodes} 个节点、{max_edges} 条边；
5. 无法确定的实体宁可不输出。

文档片段：
{text}
"""


def _dedup_key(label: str) -> str:
    return re.sub(r"\s+", "", label).lower()


def _heuristic(text: str, doc_id: str, filename: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    root_id = f"{doc_id}:root"
    nodes: list[GraphNode] = [
        GraphNode(
            id=root_id,
            label=filename,
            type="Document",
            source_doc_id=doc_id,
            properties={"chars": len(text)},
        )
    ]
    edges: list[GraphEdge] = []
    seen: dict[str, str] = {}  # dedup_key -> node_id

    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    # 段落 → 该段命中的节点 id（用于共现建边）
    para_hits: list[list[str]] = []

    for para in paragraphs:
        hits: list[str] = []
        for ntype, pattern in _PATTERNS:
            for m in re.finditer(pattern, para):
                label = m.group(0).strip()
                if len(label) < 2:
                    continue
                key = _dedup_key(label)
                nid = seen.get(key)
                if nid is None:
                    if len(nodes) >= MAX_NODES:
                        continue
                    nid = f"{doc_id}:n{len(nodes)}"
                    seen[key] = nid
                    nodes.append(
                        GraphNode(
                            id=nid,
                            label=label,
                            type=ntype,
                            source_doc_id=doc_id,
                        )
                    )
                    edges.append(
                        GraphEdge(
                            id=f"{doc_id}:e{len(edges)}",
                            source=root_id,
                            target=nid,
                            label="提及",
                            source_doc_id=doc_id,
                        )
                    )
                if nid not in hits:
                    hits.append(nid)
        para_hits.append(hits)

    # 同段共现 → 异类实体之间建「相关」边（同类不建，避免枚举噪声）
    type_of = {n.id: n.type for n in nodes}
    linked: set[tuple[str, str]] = set()
    for hits in para_hits:
        for i in range(len(hits)):
            for j in range(i + 1, len(hits)):
                a, b = hits[i], hits[j]
                if type_of.get(a) == type_of.get(b):
                    continue
                pair = (a, b) if a < b else (b, a)
                if pair in linked or len(edges) >= MAX_EDGES:
                    continue
                linked.add(pair)
                edges.append(
                    GraphEdge(
                        id=f"{doc_id}:e{len(edges)}",
                        source=pair[0],
                        target=pair[1],
                        label="相关",
                        source_doc_id=doc_id,
                    )
                )
    return nodes, edges[:MAX_EDGES]


def _parse_llm_json(raw: str) -> Optional[dict]:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _from_llm(
    text: str, doc_id: str, filename: str, model
) -> Optional[tuple[list[GraphNode], list[GraphEdge]]]:
    prompt = _LLM_PROMPT.format(
        types="/".join(NODE_TYPES),
        max_nodes=MAX_NODES,
        max_edges=MAX_EDGES,
        text=text[:4000],
    )
    try:
        raw = model.generate(prompt)
    except Exception:
        return None
    data = _parse_llm_json(raw or "")
    if not data:
        return None

    root_id = f"{doc_id}:root"
    nodes = [
        GraphNode(id=root_id, label=filename, type="Document", source_doc_id=doc_id)
    ]
    by_label: dict[str, str] = {}
    for item in (data.get("nodes") or [])[:MAX_NODES]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label or label not in text:  # 反幻觉：实体必须原文出现
            continue
        key = _dedup_key(label)
        if key in by_label:
            continue
        ntype = str(item.get("type") or "Concept")
        if ntype not in NODE_TYPES:
            ntype = "Concept"
        nid = f"{doc_id}:n{len(nodes)}"
        by_label[key] = nid
        nodes.append(
            GraphNode(id=nid, label=label, type=ntype, source_doc_id=doc_id)
        )
    if len(nodes) <= 1:
        return None  # LLM 未抽到有效实体，交回启发式

    edges = [
        GraphEdge(
            id=f"{doc_id}:e{i}",
            source=root_id,
            target=n.id,
            label="提及",
            source_doc_id=doc_id,
        )
        for i, n in enumerate(nodes[1:])
    ]
    for item in (data.get("edges") or []):
        if not isinstance(item, dict) or len(edges) >= MAX_EDGES:
            continue
        s = by_label.get(_dedup_key(str(item.get("source") or "")))
        t = by_label.get(_dedup_key(str(item.get("target") or "")))
        if not s or not t or s == t:
            continue
        edges.append(
            GraphEdge(
                id=f"{doc_id}:e{len(edges)}",
                source=s,
                target=t,
                label=str(item.get("label") or "相关")[:20],
                source_doc_id=doc_id,
            )
        )
    return nodes, edges[:MAX_EDGES]


def extract(
    text: str,
    doc_id: str,
    filename: str,
    *,
    mode: str = "none",
    model=None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """抽取图谱；``mode="auto"`` 优先 LLM，失败静默回退启发式。"""
    if mode == "auto":
        if model is None:
            try:
                from app.model import get_model

                model = get_model()
            except Exception:
                model = None
        if model is not None:
            got = _from_llm(text, doc_id, filename, model)
            if got:
                return got
    return _heuristic(text, doc_id, filename)
