"""06-知识库包入口：工厂 + 便捷函数 + Passage 契约导出。

架构（按约定「文件放 SQLite、向量检索放 DuckDB」）：
- ``kb.engine=duckdb``（默认）：SQLite 持久化文件（``kb.path`` 指向 .db） + DuckDB
  ``ATTACH`` 挂载做向量检索；
- ``kb.engine=sqlite``：纯 SQLite 回落（向量检索用 Python 余弦，无 DuckDB）。

工厂按 engine 惰性导入对应实现，默认 duckdb 路径不触发 torch 加载，
满足「无 torch 单测」验收基线。

设计文档关联：06 技术文档 §2（包结构）、§8（配置）、§7.2（便捷函数）。
"""
from __future__ import annotations

import os
from typing import Optional

from app.db.knowledge.chunking import Passage

_STORE = None


def get_kb_config() -> dict:
    """返回 kb 配置（合并默认值），缺段时返回默认 duckdb 配置。

    ``app.config`` 在此惰性导入，使 import ``app.db.knowledge.store`` 跑 DuckDB 单测时
    不触发配置加载（无 torch / 无 pyyaml 亦可）。
    """
    from app.config import config

    kb = dict(config.get("kb") or {})
    kb.setdefault("engine", "duckdb")
    kb.setdefault("path", "./kb/knowledge.db")
    kb.setdefault("embed_dim", 0)
    kb.setdefault("use_vss", False)
    chunk = dict(kb.get("chunk") or {})
    chunk.setdefault("max_chars", 600)
    chunk.setdefault("overlap", 80)
    kb["chunk"] = chunk
    return kb


def get_knowledge_store():
    """按 config.kb.engine 返回知识库实例（懒加载，单例缓存）。

    - duckdb（默认）：SQLite 文件落地 + DuckDB 向量检索；
    - sqlite：纯 SQLite 文件 + Python 余弦检索回落。
    """
    global _STORE
    if _STORE is not None:
        return _STORE
    kb = get_kb_config()
    engine = kb["engine"]
    path = kb["path"]
    kb_dir = os.path.dirname(os.path.abspath(path))
    if engine == "sqlite":
        from app.db.knowledge.sqlite_store import SQLiteKnowledgeStore

        _STORE = SQLiteKnowledgeStore(db_path=path, kb_dir=kb_dir)
    else:
        from app.db.knowledge.store import DuckDBKnowledgeStore

        _STORE = DuckDBKnowledgeStore(
            sqlite_path=path,
            dim=int(kb["embed_dim"] or 0),
            use_vss=bool(kb["use_vss"]),
            chunk_max_chars=int(kb["chunk"]["max_chars"]),
            chunk_overlap=int(kb["chunk"]["overlap"]),
            kb_dir=kb_dir,
        )
    return _STORE


def reset_knowledge_store() -> None:
    """清空单例缓存（测试用）。"""
    global _STORE
    _STORE = None


# —— 便捷函数（02 run() 经此取片段，向后兼容 04 retrieve 签名） —— #
def retrieve(query: str, top_k: int = 3) -> list[Passage]:
    return get_knowledge_store().retrieve(query, top_k=top_k)


def ingest_text(text: str, **meta) -> int:
    return get_knowledge_store().ingest_text(text, **meta)


def ingest_document(path: str, **meta) -> int:
    return get_knowledge_store().ingest_document(path, **meta)
