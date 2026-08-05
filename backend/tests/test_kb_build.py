"""06 知识库构建流程单测（无 torch 可跑：embedder 注入 mock）。

验证 `rebuild(rescan=True)` 扫描 kb/ 目录入库，且幂等（重复构建不翻倍）。
"""
from __future__ import annotations

import os
import tempfile

from app.db.knowledge.store import DuckDBKnowledgeStore

DIM = 8


class SimpleEmbedder:
    def embed(self, texts):
        return [[0.1] * DIM for _ in texts]


def _make_kb_dir(tmp: str) -> str:
    kb_dir = os.path.join(tmp, "kb")
    os.makedirs(kb_dir, exist_ok=True)
    with open(os.path.join(kb_dir, "faq.md"), "w", encoding="utf-8") as f:
        f.write("---\ntitle: FAQ\ndoc_type: product\ndomain: Banking\n---\n# 银行\n银行理财风险提示内容说明。\n")
    with open(os.path.join(kb_dir, "reg.md"), "w", encoding="utf-8") as f:
        f.write("---\ntitle: REG\ndoc_type: regulation\ndomain: Securities\n---\n# 证券\n证券创业板交易条款说明。\n")
    return kb_dir


def test_build_rescan_idempotent():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "kb.db")
    kb_dir = _make_kb_dir(tmp)
    store = DuckDBKnowledgeStore(sqlite_path=db_path, embedder=SimpleEmbedder(), kb_dir=kb_dir)

    store.rebuild(rescan=True)
    assert store.stats()["doc_count"] == 2, "应入库 2 个文档"

    # 幂等：再次 rebuild(rescan=True) 应先删后写，文档数不翻倍
    store.rebuild(rescan=True)
    assert store.stats()["doc_count"] == 2, "重复构建不应翻倍"

    docs = {d["doc_id"] for d in store.list_docs()}
    assert docs == {"faq.md", "reg.md"}
    store.close()
