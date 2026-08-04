"""06 共享文本处理工具（纯 Python，零重依赖，可在无 torch 环境导入）。

从 04 ``app.agent.retrieval`` 抽离的「切分 / BM25 / 归一化 / Passage 契约」，
供 06 DuckDB 后端与 04 SQLite 后端共用单一来源，避免重复实现、也避免 06 在
导入时被迫拉起 ``app.model``（其 local 实现依赖 torch）。

设计文档关联：06 §1.1（复用 04 的切分/BM25/Passage）、§9.2（下沉共享
``app/kb/chunking.py`` 以满足「无 torch 单测」验收基线）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


class InconsistentDimensionError(ValueError):
    """同一知识库内 embedding 维度不一致（向量检索要求等长）。"""


@dataclass
class Passage:
    """一条检索片段（02 既有接口，新增字段带默认值以兼容）。

    06 与 04 共用此契约；``chapter`` / ``effective_date`` 以及 ``title`` / ``agency`` /
    ``domain`` / ``doc_type`` / ``version`` / ``doc_id`` 用于下游总结智能体标注来源与引用。
    """

    content: str
    source: str = ""
    score: float = 0.0
    chapter: str = ""
    effective_date: str = ""
    title: str = ""
    agency: str = ""
    doc_type: str = ""
    domain: str = ""
    version: str = ""
    doc_id: str = ""


def format_passage(p: "Passage", idx: int = 0) -> str:
    """把一条 Passage 格式化为带完整溯源标记的片段，供下游总结智能体引用。

    仅展示非空字段，相关度保留 2 位小数；返回 ``[idx] 来源｜机构｜领域｜生效｜版本｜相关度``
    头 + 章节 + 正文的结构化文本。
    """
    bits: list[str] = []
    if p.title:
        bits.append(f"来源《{p.title}》")
    elif p.source:
        bits.append(f"来源：{p.source}")
    if p.agency:
        bits.append(f"机构：{p.agency}")
    if p.domain:
        bits.append(f"领域：{p.domain}")
    if p.effective_date:
        bits.append(f"生效：{p.effective_date}")
    if p.version and p.version != "1":
        bits.append(f"版本：{p.version}")
    if p.score:
        try:
            bits.append(f"相关度：{float(p.score):.2f}")
        except (TypeError, ValueError):
            bits.append(f"相关度：{p.score}")
    head = "｜".join(bits)
    block = f"[{idx}] {head}" if head else f"[{idx}]"
    if p.chapter:
        block += f"\n章节：{p.chapter}"
    block += f"\n{p.content}"
    return block


def format_passages(passages: list["Passage"], *, start: int = 1) -> str:
    """把检索结果格式化为供总结智能体消费的查询上下文；空结果返回提示语。"""
    if not passages:
        return "（暂无检索结果）"
    return "\n\n".join(format_passage(p, i) for i, p in enumerate(passages, start))


class _Chunk:
    __slots__ = ("text", "chapter")

    def __init__(self, text: str, chapter: str = "") -> None:
        self.text = text
        self.chapter = chapter


def _split_into_chunks(text: str, max_chars: int = 600, overlap: int = 80) -> list[_Chunk]:
    """按标题/段落边界切分，长块带父标题上下文，保留结构。

    - 以标题（``#`` 标题或冒号结尾短标签，如「第一条：」）分节，节内文本带「父标题」上下文；
    - 超长节按段落聚合，超过 ``max_chars`` 的块保留尾部 ``overlap`` 字符重叠；
    - 表格/短段不单独打散，保持语义完整（设计 §5.3）。
    """
    text = text.replace("\r\n", "\n")

    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        # 标题启发式：markdown 标题（``#`` 前缀）或冒号结尾的短标签。
        # 不以句号结尾的整句误判为标题（会吞掉正文），故句号结尾不视为标题。
        is_heading = bool(re.match(r"^#{1,4}\s", stripped)) or (
            stripped and len(stripped) <= 30 and stripped.endswith(("：", ":"))
        )
        if is_heading:
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
                buf = []
            heading = stripped
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))

    chunks: list[_Chunk] = []
    for hd, body in sections:
        if not body:
            continue
        if len(body) <= max_chars:
            chunks.append(_Chunk((hd + "\n" + body) if hd else body, hd))
            continue
        paras = [p for p in re.split(r"\n{1,}", body) if p.strip()]
        cur = ""
        for p in paras:
            if len(cur) + len(p) + 1 <= max_chars:
                cur = (cur + "\n" + p).strip() if cur else p
            else:
                if cur:
                    chunks.append(_Chunk((hd + "\n" + cur) if hd else cur, hd))
                cur = (p[-overlap:] + "\n" + p) if overlap else p
        if cur:
            chunks.append(_Chunk((hd + "\n" + cur) if hd else cur, hd))
    return chunks


class _BM25:
    """Okapi BM25，纯 Python 实现。

    中文按字、英文按词（``\\w`` 在 Unicode 下含汉字）。生产环境可换 jieba 分词以
    提升中文关键词召回；本期实现聚焦可用性与零额外依赖。
    """

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [self._tok(d) for d in corpus]
        self.N = len(self.docs)
        self.df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 0.0

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def scores(self, query: str) -> list[float]:
        q = self._tok(query)
        if not q:
            return [0.0] * self.N
        out = []
        for d in self.docs:
            dl = len(d)
            s = 0.0
            counts: dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            for t in q:
                if t in self.df:
                    f = counts.get(t, 0)
                    idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * (f * (self.k1 + 1)) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    )
            out.append(s)
        return out


def _normalize(xs: Sequence[float]) -> list[float]:
    """min-max 归一化到 [0, 1]；全相等时全 1（正数）或全 0（全 0）。"""
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [1.0] * len(xs) if hi > 0 else [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]
