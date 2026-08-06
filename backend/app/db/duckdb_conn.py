"""DuckDB → SQLite 挂载连接的进程内共享注册表。

背景（评审 B1「存储收口主库」）：06 知识库与 08 个人文档均把数据落在 **同一个**
SQLite 主库（``db.sqlite_path``，默认 ``agentar.db``），再由 DuckDB ``ATTACH`` 该文件做
``list_cosine_distance`` 向量检索。若两个模块各开一条 DuckDB 连接以 READ_WRITE 挂载
同一文件，会出现「双写者」竞争（08 技术文档 §13 风险表首行）。

本模块把 ``(abspath, alias)`` 作为键缓存连接，使同库同别名的调用方复用**同一条**
DuckDB 连接：

- 写入天然串行（单连接），规避多写者锁竞争；
- 06 ``kb.chunks`` 与 08 ``kb.personal_doc_chunk_embeddings`` 在同一 catalog 下，
  RAG 可在一次查询内同时覆盖「公共知识库 + 个人文档」两个来源（评审 B2）。

单测里各用例用独立临时文件路径，故互不复用；``close(path, alias)`` 会关闭并逐出缓存。
"""
from __future__ import annotations

import os
import sqlite3
import threading

import duckdb

_LOCK = threading.RLock()
_CONNS: dict[tuple[str, str], "duckdb.DuckDBPyConnection"] = {}


def _key(sqlite_path: str, alias: str) -> tuple[str, str]:
    return (os.path.abspath(sqlite_path), alias)


def get_attached(sqlite_path: str, alias: str = "kb"):
    """返回挂载了 ``sqlite_path`` 的共享 DuckDB 连接（按 (路径, alias) 缓存）。

    文件不存在时先建空 SQLite 库（DuckDB 的 sqlite 扫描器要求文件已存在）。
    """
    k = _key(sqlite_path, alias)
    with _LOCK:
        conn = _CONNS.get(k)
        if conn is not None:
            return conn
        parent = os.path.dirname(os.path.abspath(sqlite_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(sqlite_path):
            sqlite3.connect(sqlite_path).close()
        conn = duckdb.connect(":memory:")
        conn.execute(f"ATTACH '{sqlite_path}' AS {alias} (TYPE sqlite, READ_WRITE)")
        _CONNS[k] = conn
        return conn


def close(sqlite_path: str, alias: str = "kb") -> None:
    """关闭并逐出缓存连接（测试清理用；生产随进程退出）。"""
    k = _key(sqlite_path, alias)
    with _LOCK:
        conn = _CONNS.pop(k, None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def close_all() -> None:
    """关闭全部缓存连接（测试 teardown 用）。"""
    with _LOCK:
        conns = list(_CONNS.values())
        _CONNS.clear()
    for c in conns:
        try:
            c.close()
        except Exception:
            pass
