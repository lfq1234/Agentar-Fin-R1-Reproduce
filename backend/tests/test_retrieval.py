"""04 扩展：RAG 检索管线测试。

用确定性 fake embedder（不依赖真实 API/key），验证：
- 切分保留标题上下文；
- ingest + search 相关性（混合检索能召回并优先返回相关块）；
- 领域过滤、生效日过滤；
- 空库返回空；
- 联网新闻占位；
- 模块级 `retrieve` 经 02 的 tools 路径可用。
"""
from __future__ import annotations

from app.agent.retrieval import (
    NewsItem,
    Passage,
    Retriever,
    _split_into_chunks,
    news_search,
    retrieve,
)


class _FakeEmbedder:
    """确定性、可复现的 embedder：按字符哈希到固定维向量（纯 stub）。"""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for ch in text:
            h = (ord(ch) * 31) % self.dim
            v[h] += 1.0
        return v


def _retriever() -> Retriever:
    return Retriever(embedder=_FakeEmbedder())


def test_split_preserves_heading():
    text = "# 存款保险\n存款保险限额 50 万。\n# 贷款利率\nLPR 浮动。"
    chunks = _split_into_chunks(text)
    chapters = {c.chapter for c in chunks}
    assert any("存款保险" in c for c in chapters)
    assert any("贷款利率" in c for c in chapters)


def test_ingest_and_search_relevant_ranking():
    r = _retriever()
    r.ingest_text("存款保险限额 50 万，同一存款人合并计算。", doc_id="d1", domain="Banking")
    r.ingest_text("股票交易印花税为成交金额的 0.05%。", doc_id="d2", domain="Securities")
    # 不指定 domain，混合检索应把最相关的「存款保险」块排到首位。
    hits = r.search("存款保险保多少", top_k=2)
    assert hits
    assert "50" in hits[0].content


def test_domain_filter():
    r = _retriever()
    r.ingest_text("存款保险限额 50 万。", doc_id="b", domain="Banking")
    r.ingest_text("股票涨跌停限制为 10%。", doc_id="s", domain="Securities")
    banking = r.search("限额", domain="Banking")
    assert banking
    assert all("存款保险" in h.content for h in banking)
    sec = r.search("限额", domain="Securities")
    assert all("涨跌停" in h.content for h in sec)


def test_effective_date_filter():
    r = _retriever()
    r.ingest_text("旧规：利率上限 0.1%。", doc_id="old", domain="Banking",
                  effective_date="2010-01-01")
    r.ingest_text("新规：利率市场化。", doc_id="new", domain="Banking",
                  effective_date="2020-01-01")
    hits = r.search("利率", domain="Banking", effective_after="2015-01-01")
    assert hits
    assert all(h.effective_date >= "2015-01-01" for h in hits)
    assert all("新规" in h.content for h in hits)


def test_empty_returns_nothing():
    r = _retriever()
    assert r.search("anything") == []
    assert r.retrieve("anything") == []


def test_news_stub_empty():
    assert news_search("近期新规") == []


def test_news_provider():
    def prov(q, k):
        return [NewsItem(summary=f"{q} 要点", source="新华网", published_at="2026-01-01")]

    out = news_search("存款利率", provider=prov)
    assert len(out) == 1
    assert out[0].source == "新华网"


def test_module_retrieve_via_tools_path():
    # 模块级 retrieve 经 02 的 tools.lookup_knowledge 调用；空库返回空（向后兼容）。
    results: list[Passage] = retrieve("测试查询")
    assert results == []
