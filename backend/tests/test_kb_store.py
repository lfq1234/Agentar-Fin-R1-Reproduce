"""06 知识库 DuckDB 后端单测（无 torch 可跑：embedder 注入 mock）。

覆盖：ingest / 双路独立召回 + 并集融合（评审 B1）/ 元数据过滤（domain、effective_date）
/ 生命周期（stats、list_docs、delete_doc、rebuild）/ 维度不一致报错。

SQLite 后端测试（`test_sqlite_backend_fallback`）在缺 torch 时自动 skip，不阻断 DuckDB 单测。
"""
from __future__ import annotations

import os
import tempfile

import pytest

from app.kb.chunking import Passage
from app.kb.store import DuckDBKnowledgeStore, InconsistentDimensionError, _Row

DIM = 8


class MarkerEmbedder:
    """确定性 mock embedding：含 ``MARKER_XYZ`` 的文本与 query 向量正交（向量路漏召），
    其余文本与 query 同向量（向量路高相似）。用于验证 BM25 从全库补回漏召块。"""

    def embed(self, texts):
        out = []
        for t in texts:
            if "MARKER_XYZ" in t:
                out.append([1.0] + [0.0] * (DIM - 1))
            else:
                out.append([0.0] * (DIM - 1) + [1.0])
        return out


class SimpleEmbedder:
    """所有文本返回相同向量（dim=8），用于不关心向量质量的过滤/生命周期测试。"""

    def embed(self, texts):
        return [[0.1] * DIM for _ in texts]


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    s = DuckDBKnowledgeStore(sqlite_path=path, embedder=MarkerEmbedder(), kb_dir=None)
    yield s
    s.close()
    if os.path.exists(path):
        os.remove(path)


def test_ingest_returns_chunk_count(store):
    n = store.ingest_text(
        "普通金融条款内容描述。 " * 20 + "\n# 特殊\nMARKER_XYZ 创业板开通条件要求资产日均十万。",
        doc_id="d1",
        domain="Securities",
    )
    assert n > 0


def test_union_recall_merges_two_routes(store):
    """单元验证：双路独立召回后按 id 去重取并集（评审 B1 核心）。"""
    r1 = _Row(id=1, doc_id="d", title="t", agency="", doc_type="", domain="",
              version="", chapter="", content="vec-block", effective_date="")
    r2 = _Row(id=2, doc_id="d", title="t", agency="", doc_type="", domain="",
              version="", chapter="", content="bm-block", effective_date="")
    out = DuckDBKnowledgeStore._union_recall([(r1, 0.9, None)], [(r2, None, 0.8)])
    assert len(out) == 2
    merged = DuckDBKnowledgeStore._union_recall([(r1, 0.9, None)], [(r1, None, 0.8)])
    assert len(merged) == 1
    assert merged[0][1] == 0.9 and merged[0][2] == 0.8


def test_passage_carries_citation_metadata(store):
    """检索产物应透传机构/领域/标题等溯源字段，供下游总结智能体引用（而非仅 content）。"""
    store.ingest_text(
        "银行理财风险提示内容说明。 " * 5,
        doc_id="b1",
        title="商业银行理财业务管理办法",
        agency="国家金融监督管理总局",
        domain="Banking",
        doc_type="regulation",
        effective_date="2024-01-01",
        version="2",
    )
    res = store.search("银行理财风险提示", top_k=3)
    assert res, "应有结果"
    p = res[0]
    assert p.agency == "国家金融监督管理总局"
    assert p.domain == "Banking"
    assert p.title == "商业银行理财业务管理办法"
    assert p.version == "2"
    # 格式化输出应包含机构与标题，便于总结智能体引用
    from app.kb.chunking import format_passage

    text = format_passage(p, 1)
    assert "国家金融监督管理总局" in text
    assert "商业银行理财业务管理办法" in text
    assert "Banking" in text


def test_bm25_supplements_vector_recall(store):
    """向量路漏召的块（sim=0）应由 BM25 从全库独立召回补回（并集进入候选）。

    降低 vec_weight 让 BM25 主导，验证 marker 块能进入最终结果——若 BM25 被限制在
    向量候选子集（旧伪代码缺陷），该块根本不会出现在候选集，无法进结果。
    """
    base = "普通金融条款内容描述股票基金理财保险信托证券银行。 " * 300
    text = base + "\n# 特殊条款\nMARKER_XYZ 创业板开通条件要求资产日均十万且交易满二十四个月。"
    store.ingest_text(text, doc_id="d1", domain="Securities")
    res = store.search("创业板开通条件", top_k=3, vec_weight=0.1)
    assert any("MARKER_XYZ" in p.content for p in res), "BM25 应从全库补回向量漏召的块"


def test_domain_filter(store):
    store.ingest_text("银行理财风险提示内容说明。 " * 10, doc_id="b", domain="Banking")
    store.ingest_text("证券创业板交易条款说明。 " * 10, doc_id="s", domain="Securities")
    res = store.search("风险提示", top_k=10, domain="Banking")
    assert res, "应有结果"
    for p in res:
        assert "银行" in p.content, "domain=Banking 不应返回证券块"


def test_effective_after_filter(store):
    store.ingest_text("旧条款内容说明。 " * 5, doc_id="old", domain="Banking", effective_date="2020-01-01")
    store.ingest_text("新条款内容说明。 " * 5, doc_id="new", domain="Banking", effective_date="2024-01-01")
    res = store.search("条款", top_k=10, effective_after="2023-01-01")
    assert res, "应有结果"
    for p in res:
        assert "新条款" in p.content, "effective_after 应排除 2020 旧条款"


def test_empty_store_returns_empty(store):
    assert store.search("anything") == []
    assert store.retrieve("anything") == []


def test_stats(store):
    store.ingest_text("内容A说明。 " * 5, doc_id="a")
    st = store.stats()
    assert st["doc_count"] == 1
    assert st["chunk_count"] > 0
    assert st["dim"] == DIM


def test_delete_and_list_docs(store):
    store.ingest_text("内容A说明。 " * 5, doc_id="a", domain="Banking")
    store.ingest_text("内容B说明。 " * 5, doc_id="b", domain="Securities")
    assert {d["doc_id"] for d in store.list_docs()} == {"a", "b"}
    n = store.delete_doc("a")
    assert n >= 1
    assert {d["doc_id"] for d in store.list_docs()} == {"b"}


def test_rebuild_clears(store):
    store.ingest_text("内容A说明。 " * 5, doc_id="a")
    deleted = store.rebuild()
    assert deleted >= 1
    assert store.list_docs() == []
    assert store.stats()["chunk_count"] == 0


def test_dimension_inconsistency(store):
    store.ingest_text("内容A说明。 " * 5, doc_id="a")  # dim = DIM

    class BadEmbedder:
        def embed(self, texts):
            return [[0.0] * 4 for _ in texts]  # dim = 4，不一致

    store.set_embedder(BadEmbedder())
    with pytest.raises(InconsistentDimensionError):
        store.ingest_text("内容B说明。 " * 5, doc_id="b")


def test_sqlite_backend_fallback():
    """SQLite 双后端回落（仅在 torch 可用时运行；无 torch 自动 skip）。"""
    sqlite_store = pytest.importorskip("app.kb.sqlite_store")
    s = sqlite_store.SQLiteKnowledgeStore(db_path=":memory:", kb_dir=None)
    s.set_embedder(SimpleEmbedder())
    n = s.ingest_text("银行理财风险提示内容说明。 " * 5, doc_id="b", domain="Banking")
    assert n > 0
    res = s.search("风险提示", top_k=3, domain="Banking")
    assert all("银行" in p.content for p in res)
