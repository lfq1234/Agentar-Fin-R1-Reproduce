"""04 扩展：RAG 本地文档管线（在 02 既有 `Passage` / `retrieve()` 之上扩展）。

保留 02 既有接口不变：
- `Passage`（增加可选字段 `chapter` / `effective_date`，默认值兼容既有调用）；
- `retrieve(query, top_k)`：02 的 `tools.py` 经此取检索片段，行为向后兼容
  （无索引时返回空列表，工具层照常输出「暂无检索结果」）。

新增能力（评审 G1 闭环，不新建包）：
- 接入/解析/切分（ingest_document / ingest_text + `_split_into_chunks`）；
- 向量化（经 01 的 `get_embedder()`，api/local 双模式）；
- 存储（本地 SQLite，chunks + 向量 + 元数据）；
- 混合检索（向量余弦 + BM25）+ 元数据过滤 + 重排（`search`）；
- 联网新闻占位（`news_search`，真实 API 为后续需求，提供可插拔 provider）。

定位：把本地金融文档转为可检索、可追溯的知识，供信息查询智能体（rag 角色）消费。
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from app.model import get_embedder


@dataclass
class Passage:
    """一条检索片段（02 既有接口，新增字段带默认值以兼容）。"""

    content: str
    source: str = ""
    score: float = 0.0
    chapter: str = ""
    effective_date: str = ""


@dataclass
class NewsItem:
    """一条联网新闻摘要（04 外联新闻管线输出）。"""

    summary: str
    source: str = ""
    published_at: str = ""
    url: str = ""
    credibility: float = 0.0


# --------------------------------------------------------------------------- #
# 切分（chunking）
# --------------------------------------------------------------------------- #

class _Chunk:
    __slots__ = ("text", "chapter")

    def __init__(self, text: str, chapter: str = "") -> None:
        self.text = text
        self.chapter = chapter


def _split_into_chunks(text: str, max_chars: int = 600, overlap: int = 80) -> list[_Chunk]:
    """按标题/段落边界切分，长块带父标题上下文，保留结构。

    - 以标题（`#` 标题或短句结尾标点）分节，节内文本带「父标题」上下文；
    - 超长节按段落聚合，超过 `max_chars` 的块保留尾部 `overlap` 字符重叠；
    - 表格/短段不单独打散，保持语义完整（设计 §5.3）。
    """
    text = text.replace("\r\n", "\n")

    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        # 标题启发式：markdown 标题（`#` 前缀）或冒号结尾的短标签（如「第一条：」）。
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


# --------------------------------------------------------------------------- #
# BM25（轻量关键词召回，与向量余弦融合）
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# 数学工具
# --------------------------------------------------------------------------- #

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _normalize(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [1.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


# --------------------------------------------------------------------------- #
# 检索器（核心）
# --------------------------------------------------------------------------- #

class Retriever:
    """本地 RAG 检索器：接入 → 切分 → 向量化 → 存储 → 混合检索。

    依赖 01 的 `get_embedder()` 完成向量化（api/local 双模式），懒加载，不在
    构造或导入时触发真实模型请求。无索引时 `search`/`retrieve` 返回空列表。
    """

    def __init__(self, db_path: str = ":memory:", embedder=None) -> None:
        self._db_path = db_path
        self._embedder = embedder  # 可注入（测试用）；为 None 时懒加载 01
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "doc_id TEXT, title TEXT, doc_type TEXT, agency TEXT,"
            "domain TEXT, chapter TEXT, effective_date TEXT,"
            "content TEXT, embedding TEXT)"
        )
        self._conn.commit()

    # —— embedder —— #
    def set_embedder(self, fn) -> None:
        """注入 embedding 函数（测试或自定义实现用）。"""
        self._embedder = fn

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()  # 01 扩展，懒加载
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().embed(texts)

    # —— ingest（接入 / 解析 / 切分 / 向量化 / 存储） —— #
    def ingest_text(
        self,
        text: str,
        *,
        doc_id: str = "doc",
        title: str = "",
        doc_type: str = "internal",
        agency: str = "",
        domain: str = "",
        effective_date: str = "",
        max_chars: int = 600,
        overlap: int = 80,
    ) -> int:
        """将一段文本切分、向量化并写入索引；返回写入块数。"""
        chunks = _split_into_chunks(text, max_chars=max_chars, overlap=overlap)
        if not chunks:
            return 0
        vecs = self._embed([c.text for c in chunks])
        cur = self._conn.cursor()
        for c, v in zip(chunks, vecs):
            cur.execute(
                "INSERT INTO chunks "
                "(doc_id,title,doc_type,agency,domain,chapter,effective_date,content,embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    title,
                    doc_type,
                    agency,
                    domain,
                    c.chapter,
                    effective_date,
                    c.text,
                    json.dumps(v, ensure_ascii=False),
                ),
            )
        self._conn.commit()
        return len(chunks)

    def ingest_document(self, path: str, **meta) -> int:
        """按扩展名接入文档；txt/md/html 原生支持，二进制需可选依赖。"""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".txt", ".md", ".markdown"):
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            title = meta.pop("title", os.path.basename(path))
            doc_id = meta.pop("doc_id", os.path.basename(path))
            return self.ingest_text(text, doc_id=doc_id, title=title, **meta)
        if ext in (".html", ".htm"):
            with open(path, encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            text = re.sub(r"<script[\s\S]*?</script>", " ", raw)
            text = re.sub(r"<style[\s\S]*?</style>", " ", text)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&nbsp;", " ", text)
            text = re.sub(r"\s+", " ", text)
            title = meta.pop("title", os.path.basename(path))
            doc_id = meta.pop("doc_id", os.path.basename(path))
            return self.ingest_text(text, doc_id=doc_id, title=title, **meta)
        # 二进制格式需可选依赖；本期提供扩展点，缺失时给出清晰指引。
        raise NotImplementedError(
            f"暂不支持 {ext} 解析；PDF/DOCX/XLSX 需安装 pdfminer.six / python-docx / openpyxl，"
            f"本期 RAG 实现聚焦 txt/md/html，二进制解析为后续工程需求。"
        )

    # —— search（混合检索 + 过滤 + 重排） —— #
    def search(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
        effective_after: Optional[str] = None,
        vec_weight: float = 0.7,
    ) -> list[Passage]:
        """混合检索：向量余弦 + BM25，按元数据过滤后融合重排。

        Args:
            query: 检索查询。
            top_k: 返回条数。
            domain: 领域过滤（逗号分隔多标签，命中其一即可）。
            effective_after: 生效日下界（ISO 字符串，优先现行有效）。
            vec_weight: 向量得分在融合中的权重（BM25 权重为 1 - vec_weight）。
        """
        # 列序：id, doc_id, title, domain, chapter, content, effective_date, embedding
        rows = self._conn.execute(
            "SELECT id,doc_id,title,domain,chapter,content,effective_date,embedding "
            "FROM chunks"
        ).fetchall()
        if not rows:
            return []

        # 元数据过滤
        cands = []
        for r in rows:
            dom = (r[3] or "").strip()
            eff = (r[6] or "").strip()
            if domain:
                tags = [x.strip() for x in dom.split(",") if x.strip()]
                if domain not in tags:
                    continue
            if effective_after and eff and eff < effective_after:
                continue
            cands.append(r)
        if not cands:
            return []

        contents = [r[5] for r in cands]
        qvec = self._embed([query])[0]
        bm25 = _BM25(contents)
        bm25_scores = bm25.scores(query)
        vec_scores = [_cosine(qvec, json.loads(r[7])) for r in cands]

        nv = _normalize(vec_scores)
        nb = _normalize(bm25_scores)
        fused = [vec_weight * a + (1 - vec_weight) * b for a, b in zip(nv, nb)]
        ranked = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)[:top_k]

        out: list[Passage] = []
        for i in ranked:
            r = cands[i]
            out.append(
                Passage(
                    content=r[5],
                    source=f"{r[2] or r[1]}",
                    score=round(fused[i], 4),
                    chapter=r[4] or "",
                    effective_date=r[6] or "",
                )
            )
        return out

    def retrieve(self, query: str, top_k: int = 3) -> list[Passage]:
        """02 既有接口：返回 `list[Passage]`；无索引时恒返回空列表。"""
        return self.search(query, top_k=top_k)


# --------------------------------------------------------------------------- #
# 模块级单例与便捷函数（02 的 tools.py 经 `retrieve` 取片段）
# --------------------------------------------------------------------------- #

_DEFAULT: Optional[Retriever] = None


def _default_retriever() -> Retriever:
    global _DEFAULT
    if _DEFAULT is None:
        db = os.environ.get("AGENTAR_RAG_DB", ":memory:")
        _DEFAULT = Retriever(db_path=db)
    return _DEFAULT


def retrieve(query: str, top_k: int = 3) -> list[Passage]:
    """模块级检索入口（02 既有），向后兼容。"""
    return _default_retriever().retrieve(query, top_k=top_k)


def ingest_text(text: str, **meta) -> int:
    """模块级文本接入入口。"""
    return _default_retriever().ingest_text(text, **meta)


def ingest_document(path: str, **meta) -> int:
    """模块级文档接入入口。"""
    return _default_retriever().ingest_document(path, **meta)


def news_search(
    query: str,
    provider: Optional[Callable[[str, int], list[NewsItem]]] = None,
    top_k: int = 5,
) -> list[NewsItem]:
    """联网新闻检索占位（真实 API 为后续需求）。

    provider: 可插拔的检索实现（输入 query, top_k，输出 NewsItem 列表）；
    未提供时返回空列表（信息查询智能体据此如实声明「未检索到相关近期信息」）。
    """
    if provider is None:
        return []
    return provider(query, top_k)
