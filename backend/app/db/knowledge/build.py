"""06 知识库构建脚本：扫描 kb/ 目录语料入库，生成 DuckDB 单文件。

用法：
    python -m app.db.knowledge.build            # 读取 config.kb.path 同级 kb/ 目录，清空重建
    python -m app.db.knowledge.build --rescan   # 同上（默认即清空重建）
    python -m app.db.knowledge.build --kb-dir /path/to/kb

设计文档关联：06 技术文档 §2（build.py）、§9.6（sample 语料 + 构建脚本）。
"""
from __future__ import annotations

import argparse
import os
import sys

from app.db.knowledge import get_knowledge_store


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 06 DuckDB 知识库")
    parser.add_argument(
        "--kb-dir",
        default=None,
        help="知识库语料目录（默认取 config.kb.path 同级 kb/）",
    )
    parser.add_argument(
        "--rescan", action="store_true", help="清空后重新扫描入库（默认行为）"
    )
    args = parser.parse_args()

    store = get_knowledge_store()
    kb_dir = args.kb_dir or getattr(store, "_kb_dir", None) or "./kb"
    if not os.path.isdir(kb_dir):
        print(f"[kb-build] 知识库目录不存在: {kb_dir}", file=sys.stderr)
        print(
            "[kb-build] 请先放入语料（.md/.txt/.html），参考 backend/kb/sample_*.md",
            file=sys.stderr,
        )
        return 1

    store._kb_dir = kb_dir
    print(f"[kb-build] 扫描目录: {kb_dir}")
    deleted = store.rebuild(rescan=True)
    stats = store.stats()
    print(f"[kb-build] 已清空块数: {deleted}")
    print(f"[kb-build] 统计: {stats}")
    for d in store.list_docs():
        print(f"  - {d['doc_id']} ({d['chunk_count']} 块) domain={d['domain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
