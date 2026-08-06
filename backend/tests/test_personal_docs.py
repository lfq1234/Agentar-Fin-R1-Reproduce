"""08-个人文档与知识图谱：存储 + ingest 编排单测（无 torch 可跑，embedder 注入 mock）。

覆盖：
- ingest 全链路（解析 → 切分 → 嵌入 → 块 → 向量 → 图谱 → done）；
- 失败隔离（二进制类型 / 超限文件降级 status=error，不影响同批次其它文件、不抛异常）；
- 用户作用域（list / get / delete 跨用户不可见）；
- 图谱只暴露 status=done 的文档；
- 删除的应用层级联（文档 / 块 / 向量 / 图谱节点 / 图谱边 五处清干净）。

所有数据落 tmp_path 下的临时主库，不触碰仓库内 agentar.db。
"""
from __future__ import annotations

import pytest

from app.db import duckdb_conn
from app.db import personal_docs as pd
from app.services import documents_service as svc

DIM = 5
_KEYWORDS = ("养老", "创业板", "保险", "银行")

DOC_MD = """# 我的理财规划
工商银行 的稳健理财产品 适合养老储备，风险偏好保守。

第三条 约定 年化收益率：3.5% ，管理费率 0.3%。

风险等级 为中低，由 张三经理 负责跟进养老资金账户。
"""

DOC_OTHER = """# 创业板开户备忘
开通创业板需要资产日均十万，同时满足两年交易经验。
"""


class KeywordEmbedder:
    """确定性 mock embedding：按关键词命中置位，未命中落到兜底维（避免零向量 NaN）。"""

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIM
            for i, k in enumerate(_KEYWORDS):
                if k in t:
                    v[i] = 1.0
            if not any(v):
                v[DIM - 1] = 1.0
            out.append(v)
        return out


@pytest.fixture
def db(tmp_path, monkeypatch):
    """把 08 的主库指到临时文件，并注入 mock embedder；退出时关闭全部连接。"""
    path = str(tmp_path / "agentar_test.db")
    monkeypatch.setattr(pd, "get_main_db_path", lambda: path)
    pd.reset_personal_docs()
    svc.set_embedder(KeywordEmbedder())
    yield path
    pd.reset_personal_docs(close=True)
    duckdb_conn.close_all()
    svc.set_embedder(None)


def _ingest(name: str, text: str, user_id: int = 1) -> dict:
    return svc.ingest_one(name, text.encode("utf-8"), user_id)


# —— ingest 主流程 —— #
def test_ingest_markdown_reaches_done_with_chunks_and_vectors(db):
    rec = _ingest("理财规划.md", DOC_MD)

    assert rec["status"] == "done"
    assert rec["error"] is None
    assert rec["summary"]  # 摘要用于前端「基于此文档提问」
    assert pd.get_personal_doc_store().count_chunks(rec["id"]) > 0
    assert pd.get_vector_index().stats(user_id=1)["vector_count"] > 0
    assert pd.get_vector_index().stats(user_id=1)["dim"] == DIM


def test_ingest_html_strips_tags(db):
    raw = b"<html><body><p>\xe5\x85\xbb\xe8\x80\x81\xe4\xbf\x9d\xe9\x9a\xa9</p></body></html>"
    rec = svc.ingest_one("plan.html", raw, 1)

    assert rec["status"] == "done"
    assert "<p>" not in (rec["summary"] or "")


# —— 失败隔离 —— #
def test_unsupported_type_marks_error_not_raise(db):
    rec = svc.ingest_one("report.pdf", b"%PDF-1.4 binary", 1)

    assert rec["status"] == "error"
    assert "pdf" in rec["error"]
    # 失败文档同样进列表（前端要展示失败原因），但不产出块 / 向量
    assert pd.get_personal_doc_store().count_chunks(rec["id"]) == 0
    assert pd.get_vector_index().stats(user_id=1)["vector_count"] == 0


def test_empty_document_marks_error(db):
    rec = svc.ingest_one("empty.txt", b"   \n  \n", 1)

    assert rec["status"] == "error"
    assert rec["error"]


def test_oversize_file_marks_error(db, monkeypatch):
    cfg = dict(pd.get_personal_docs_config())
    cfg["max_file_mb"] = 0.000001  # 1 字节量级上限
    monkeypatch.setattr(svc, "get_personal_docs_config", lambda: cfg)

    rec = _ingest("big.md", DOC_MD)

    assert rec["status"] == "error"
    assert "大小上限" in rec["error"]


def test_batch_isolates_single_failure(db):
    recs = svc.ingest_files(
        [
            ("理财规划.md", DOC_MD.encode("utf-8")),
            ("report.pdf", b"%PDF-1.4"),
            ("创业板.md", DOC_OTHER.encode("utf-8")),
        ],
        user_id=1,
    )

    assert [r["status"] for r in recs] == ["done", "error", "done"]
    assert len(svc.list_documents(1)) == 3


# —— 用户作用域 —— #
def test_documents_scoped_by_user(db):
    mine = _ingest("理财规划.md", DOC_MD, user_id=1)
    _ingest("创业板.md", DOC_OTHER, user_id=2)

    assert [d["id"] for d in svc.list_documents(1)] == [mine["id"]]
    assert len(svc.list_documents(2)) == 1
    assert svc.get_status(mine["id"], user_id=2) is None  # 越权查不到


def test_delete_other_user_document_returns_false(db):
    rec = _ingest("理财规划.md", DOC_MD, user_id=1)

    assert svc.delete_document(rec["id"], user_id=2) is False
    assert svc.get_status(rec["id"], user_id=1) is not None


# —— 图谱 —— #
def test_graph_extracted_by_heuristic(db):
    rec = _ingest("理财规划.md", DOC_MD)
    nodes, edges = svc.get_graph(1)

    assert nodes and edges
    roots = [n for n in nodes if n["type"] == "Document"]
    assert len(roots) == 1 and roots[0]["label"] == "理财规划.md"
    assert {"Org", "Product"} <= {n["type"] for n in nodes}
    assert all(n["source_doc_id"] == rec["id"] for n in nodes)
    # 边两端都必须落在节点集合内（前端力导向布局要求闭合）
    ids = {n["id"] for n in nodes}
    assert all(e["source"] in ids and e["target"] in ids for e in edges)


def test_graph_excludes_failed_documents(db):
    svc.ingest_one("report.pdf", b"%PDF-1.4", 1)
    nodes, edges = svc.get_graph(1)

    assert nodes == [] and edges == []


def test_graph_scoped_by_user(db):
    _ingest("理财规划.md", DOC_MD, user_id=1)
    nodes, _ = svc.get_graph(2)

    assert nodes == []


# —— 删除级联 —— #
def test_delete_document_cascades_all_tables(db):
    rec = _ingest("理财规划.md", DOC_MD)
    doc_id = rec["id"]
    store = pd.get_personal_doc_store()
    assert store.count_chunks(doc_id) > 0

    assert svc.delete_document(doc_id, user_id=1) is True

    assert svc.list_documents(1) == []
    assert store.count_chunks(doc_id) == 0
    assert pd.get_vector_index().stats(user_id=1)["vector_count"] == 0
    assert svc.get_graph(1) == ([], [])
    assert svc.delete_document(doc_id, user_id=1) is False  # 幂等：已删再删为 False


# —— 向量检索（用户作用域） —— #
def test_vector_search_is_user_scoped(db):
    _ingest("理财规划.md", DOC_MD, user_id=1)
    _ingest("创业板.md", DOC_OTHER, user_id=2)
    qvec = KeywordEmbedder().embed(["养老"])[0]
    index = pd.get_vector_index()

    mine = index.vector_search(qvec, user_id=1, top_k=5)
    others = index.vector_search(qvec, user_id=2, top_k=5)

    assert mine and all(p.doc_type == "personal" for p in mine)
    assert "养老" in mine[0].content
    assert all(p.source == "创业板.md" for p in others)  # 只看得到自己的文档


def test_reingest_same_document_replaces_vectors(db):
    rec = _ingest("理财规划.md", DOC_MD)
    store = pd.get_personal_doc_store()
    before = pd.get_vector_index().stats(user_id=1)["vector_count"]

    # 同 doc_id 重跑（模拟重新解析）：块与向量都应是覆盖而非累加
    text = DOC_MD
    chunks = store.add_chunks(rec["id"], 1, _chunks(text))
    pd.get_vector_index().upsert(
        rec["id"], 1, chunks, KeywordEmbedder().embed([c.text for c in _chunks(text)])
    )

    assert pd.get_vector_index().stats(user_id=1)["vector_count"] == before


def _chunks(text: str):
    from app.db.knowledge.chunking import _split_into_chunks

    return _split_into_chunks(text, max_chars=600, overlap=80)
