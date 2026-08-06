"""08 × 06 × 04：RAG 联合检索与作用域打通的集成单测（无 torch 可跑）。

覆盖评审 B2「用户作用域检索」的三条链路：
1. 存储层：06 ``DuckDBKnowledgeStore.retrieve`` 能在一次检索内融合「公共知识库 +
   该用户个人文档」（同库同连接，个人文档块以合成负 id 并入 ``_union_recall``）；
2. 隔离性：``use_personal_docs=False`` 或换个 ``user_id`` 时，个人文档不得被召回；
3. 链路层：``run()`` 用 ContextVar 下传作用域，``tools.lookup_knowledge`` 读取后
   传给知识库——工具签名里没有 ``user_id``，模型无法编造他人身份（防越权）。
"""
from __future__ import annotations

import pytest

from app.agent import tools as agent_tools
from app.agent.rag_scope import current_scope, scoped
from app.db import duckdb_conn
from app.db import personal_docs as pd
from app.db.knowledge.store import DuckDBKnowledgeStore
from app.services import documents_service as svc

DIM = 5
_KEYWORDS = ("养老", "创业板", "保险", "银行")

PUBLIC_DOC = """# 养老金融监管要求
第一条 商业银行销售养老理财产品应做好投资者适当性管理。
第二条 养老理财产品不得承诺保本保收益。
"""

PERSONAL_DOC = """# 我的养老规划备忘
我在工商银行 持有一只稳健理财产品，用于养老储备。
第三条 约定 年化收益率：3.5% ，风险等级 中低。
"""

OTHER_USER_DOC = """# 创业板开户备忘
开通创业板需要资产日均十万并满足两年交易经验。
"""


class KeywordEmbedder:
    """确定性 mock embedding（与 08 单测同实现），保证检索结果可断言。"""

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
def kb(tmp_path, monkeypatch):
    """公共知识库与个人文档共用同一临时主库、同一条 DuckDB 连接（评审 B1/B2）。"""
    path = str(tmp_path / "agentar_test.db")
    monkeypatch.setattr(pd, "get_main_db_path", lambda: path)
    pd.reset_personal_docs()
    svc.set_embedder(KeywordEmbedder())

    store = DuckDBKnowledgeStore(
        sqlite_path=path, embedder=KeywordEmbedder(), kb_dir=None
    )
    monkeypatch.setattr(svc, "get_knowledge_store", lambda: store)
    store.ingest_text(
        PUBLIC_DOC, doc_id="reg-001", title="养老金融监管要求", doc_type="regulation"
    )
    svc.ingest_one("养老规划.md", PERSONAL_DOC.encode("utf-8"), 1)
    svc.ingest_one("创业板.md", OTHER_USER_DOC.encode("utf-8"), 2)

    yield store

    pd.reset_personal_docs(close=True)
    duckdb_conn.close_all()
    svc.set_embedder(None)


# —— 存储层：联合召回 —— #
def test_retrieve_unions_public_and_personal(kb):
    hits = kb.retrieve("养老", top_k=5, user_id=1, use_personal_docs=True)
    types = {p.doc_type for p in hits}

    assert "personal" in types, "个人文档未被并入召回"
    assert types - {"personal"}, "公共知识库片段被个人文档挤没了"
    assert any("养老规划.md" == p.title for p in hits)


def test_personal_docs_excluded_by_default(kb):
    hits = kb.retrieve("养老", top_k=5, user_id=1)

    assert hits, "公共知识库应仍可召回"
    assert all(p.doc_type != "personal" for p in hits)


def test_personal_docs_scoped_by_user(kb):
    mine = kb.retrieve("养老", top_k=5, user_id=1, use_personal_docs=True)
    others = kb.retrieve("养老", top_k=5, user_id=2, use_personal_docs=True)

    assert any(p.title == "养老规划.md" for p in mine)
    assert all(p.title != "养老规划.md" for p in others), "跨用户召回到了他人文档"


def test_personal_hits_carry_traceability(kb):
    hits = [
        p
        for p in kb.retrieve("养老", top_k=5, user_id=1, use_personal_docs=True)
        if p.doc_type == "personal"
    ]

    assert hits
    p = hits[0]
    assert p.doc_id and p.title == "养老规划.md" and p.source == "养老规划.md"
    assert p.score > 0


def test_service_rag_retrieve_without_user_falls_back_to_public(kb):
    hits = svc.rag_retrieve("养老", user_id=None, top_k=5)

    assert hits
    assert all(p.doc_type != "personal" for p in hits)


def test_service_rag_context_is_formatted(kb):
    ctx = svc.rag_context("养老", user_id=1, top_k=3)

    assert "养老" in ctx
    assert ctx.startswith("[1]")  # format_passages 的溯源编号


# —— 链路层：作用域透传 —— #
def test_scope_defaults_to_anonymous():
    scope = current_scope()

    assert scope.user_id is None and scope.use_personal_docs is False


def test_scope_restores_after_block():
    with scoped(user_id=7, use_personal_docs=True) as s:
        assert s.user_id == 7
        assert current_scope().user_id == 7
        assert current_scope().use_personal_docs is True

    assert current_scope().user_id is None


def test_lookup_knowledge_passes_scope_to_store(monkeypatch):
    """工具签名只有 query；user_id 必须来自作用域而非模型入参。"""
    seen: dict = {}

    class _Recorder:
        def retrieve(self, query, **kwargs):
            seen["query"] = query
            seen.update(kwargs)
            return []

    monkeypatch.setattr(agent_tools, "get_knowledge_store", lambda: _Recorder())

    with scoped(user_id=42, use_personal_docs=True):
        agent_tools.lookup_knowledge("养老理财")

    assert seen == {"query": "养老理财", "user_id": 42, "use_personal_docs": True}


def test_lookup_knowledge_outside_scope_is_anonymous(monkeypatch):
    seen: dict = {}

    class _Recorder:
        def retrieve(self, query, **kwargs):
            seen.update(kwargs)
            return []

    monkeypatch.setattr(agent_tools, "get_knowledge_store", lambda: _Recorder())
    agent_tools.lookup_knowledge("养老理财")

    assert seen == {"user_id": None, "use_personal_docs": False}


def test_tool_signature_has_no_user_id():
    """回归护栏：一旦有人把 user_id 加进工具签名，LLM 就能编造他人 id 越权。"""
    import inspect

    params = inspect.signature(agent_tools.retrieve_documents).parameters

    assert list(params) == ["query"]
