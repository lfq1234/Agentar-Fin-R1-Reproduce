"""08-个人文档与知识图谱：包入口（配置 + 单例工厂）。

层位与 06 ``app/db/knowledge/__init__.py`` 对齐：本包只做存储访问，
ingest 编排在 ``app/services/documents_service.py``，HTTP 在 ``app/routes/documents.py``。

存储收口（评审 B1）：结构化表与向量表都落 ``config.db.sqlite_path``（默认
``./agentar.db``），不另开文件；向量连接与 06 共用（``app/db/duckdb_conn.py``）。
"""
from __future__ import annotations

from typing import Optional

_STORE = None
_VECTOR = None


def get_personal_docs_config() -> dict:
    """返回 personal_docs 配置（合并默认值）。``app.config`` 惰性导入，便于无配置单测。"""
    from app.config import config

    cfg = dict(config.get("personal_docs") or {})
    cfg.setdefault("enabled", True)
    cfg.setdefault("max_file_mb", 20)
    cfg.setdefault("embed_dim", 0)
    chunk = dict(cfg.get("chunk") or {})
    chunk.setdefault("max_chars", 600)
    chunk.setdefault("overlap", 80)
    cfg["chunk"] = chunk
    graph = dict(cfg.get("graph") or {})
    graph.setdefault("enabled", True)
    graph.setdefault("model", "none")
    cfg["graph"] = graph
    rag = dict(cfg.get("rag") or {})
    rag.setdefault("enabled", True)
    rag.setdefault("top_k", 3)
    cfg["rag"] = rag
    cfg["db_path"] = get_main_db_path()
    return cfg


def get_main_db_path() -> str:
    """03 主库路径（08 不单独配置路径，永不另开文件）。"""
    from app.config import config

    return (config.get("db") or {}).get("sqlite_path") or "./agentar.db"


def get_personal_doc_store():
    """返回 :class:`PersonalDocStore` 单例（懒加载，连 03 主库）。"""
    global _STORE
    if _STORE is None:
        from app.db.personal_docs.store import PersonalDocStore

        _STORE = PersonalDocStore(get_main_db_path())
    return _STORE


def get_vector_index():
    """返回 :class:`PersonalDocVectorIndex` 单例（懒加载，DuckDB 挂载主库）。"""
    global _VECTOR
    if _VECTOR is None:
        from app.db.personal_docs.vector import PersonalDocVectorIndex

        cfg = get_personal_docs_config()
        _VECTOR = PersonalDocVectorIndex(
            cfg["db_path"], dim=int(cfg["embed_dim"] or 0)
        )
    return _VECTOR


def reset_personal_docs(close: bool = False) -> None:
    """清空单例缓存（测试用）；``close=True`` 时一并关闭底层连接。"""
    global _STORE, _VECTOR
    if close and _STORE is not None:
        try:
            _STORE.close()
        except Exception:
            pass
    _STORE = None
    _VECTOR = None
