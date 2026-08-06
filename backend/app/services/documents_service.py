"""08 个人文档与知识图谱：服务层（ingest 编排 + 生命周期 + RAG 检索）。

层位：路由（``app/routes/documents.py``）只做参数解析与序列化，业务编排在此，
数据访问在 ``app/db/personal_docs/``——与 ``chat_service`` / ``analyze_service`` 同范式。

ingest 流水线（技术文档 §6）：
    解析 → 切分（复用 06/04 chunking）→ 嵌入（复用 01）→ 写块 → 写向量
    → 抽图谱 → done

失败隔离：单文档异常只把该文档标记为 ``error``，不影响同批次其它文件、不抛 500。
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from app.db.knowledge import get_knowledge_store
from app.db.knowledge.chunking import Passage, _split_into_chunks, format_passages
from app.db.personal_docs import (
    get_personal_doc_store,
    get_personal_docs_config,
    get_vector_index,
)
from app.db.personal_docs import graph_extract, parser
from app.db.personal_docs.store import new_doc_id

logger = logging.getLogger(__name__)

_EMBEDDER = None


def set_embedder(fn) -> None:
    """注入嵌入实现（无 torch 单测用 mock；生产留空走 01 ``get_embedder()``）。"""
    global _EMBEDDER
    _EMBEDDER = fn


def _embed(texts: list[str]) -> list[list[float]]:
    global _EMBEDDER
    if _EMBEDDER is None:
        from app.model import get_embedder  # 惰性导入，避免顶层拉起 torch

        _EMBEDDER = get_embedder()
    return _EMBEDDER.embed(texts)


# —— ingest —— #
def ingest_one(filename: str, raw: bytes, user_id: int) -> dict:
    """处理单个上传文件，返回该文档的最终状态记录（含 error 时也返回记录）。"""
    cfg = get_personal_docs_config()
    store = get_personal_doc_store()
    doc_id = new_doc_id()
    size = len(raw)
    store.create_doc(doc_id, user_id, filename, size)

    max_bytes = int(cfg["max_file_mb"]) * 1024 * 1024
    try:
        if size > max_bytes:
            raise parser.UnsupportedDocument(
                f"文件超过大小上限 {cfg['max_file_mb']}MB（当前 {size / 1048576:.1f}MB）"
            )
        store.set_status(doc_id, "parsing")
        text = parser.parse_bytes(filename, raw)

        chunks = _split_into_chunks(
            text,
            max_chars=int(cfg["chunk"]["max_chars"]),
            overlap=int(cfg["chunk"]["overlap"]),
        )
        if not chunks:
            raise parser.EmptyDocument("文档切分后无有效内容块。")

        vectors = _embed([c.text for c in chunks])
        chunk_ids = store.add_chunks(doc_id, user_id, chunks)
        get_vector_index().upsert(doc_id, user_id, chunk_ids, vectors)

        if cfg["graph"]["enabled"]:
            nodes, edges = graph_extract.extract(
                text, doc_id, filename, mode=str(cfg["graph"]["model"])
            )
            store.add_graph(doc_id, user_id, nodes, edges)

        store.set_status(doc_id, "done", summary=parser.make_summary(text))
    except (parser.UnsupportedDocument, parser.EmptyDocument) as exc:
        store.set_status(doc_id, "error", error=str(exc))
    except Exception as exc:  # 嵌入 / 落库等异常同样降级为该文档 error，不 500
        logger.warning("[08] 文档处理失败 doc_id=%s file=%s: %s", doc_id, filename, exc)
        store.set_status(doc_id, "error", error=f"处理失败：{exc}")

    return store.get_doc(doc_id, user_id) or {
        "id": doc_id,
        "filename": filename,
        "size": size,
        "status": "error",
        "error": "文档记录丢失",
        "uploaded_at": "",
        "summary": None,
    }


def ingest_files(files: Sequence[tuple[str, bytes]], user_id: int) -> list[dict]:
    """批量处理 ``(filename, bytes)``；逐个隔离失败，返回本批全部文档记录。"""
    return [ingest_one(name, raw, user_id) for name, raw in files]


# —— 生命周期 —— #
def list_documents(user_id: int) -> list[dict]:
    return get_personal_doc_store().list_docs(user_id)


def get_status(doc_id: str, user_id: int) -> Optional[dict]:
    return get_personal_doc_store().get_doc(doc_id, user_id)


def delete_document(doc_id: str, user_id: int) -> bool:
    """删除文档：先清向量（DuckDB 连接），再应用层级联清四张结构化表。"""
    store = get_personal_doc_store()
    if store.get_doc(doc_id, user_id) is None:
        return False
    try:
        get_vector_index().delete_doc(doc_id)
    except Exception as exc:  # 向量清理失败不应阻断结构化删除，记日志后继续
        logger.warning("[08] 向量清理失败 doc_id=%s: %s", doc_id, exc)
    return store.delete_document(doc_id, user_id)


def get_graph(user_id: int) -> tuple[list[dict], list[dict]]:
    return get_personal_doc_store().get_graph(user_id)


# —— RAG 检索（04 智能体与自检端点共用） —— #
def rag_retrieve(
    query: str,
    *,
    user_id: Optional[int] = None,
    top_k: Optional[int] = None,
    use_personal_docs: bool = True,
) -> list[Passage]:
    """联合检索「公共知识库 + 该用户个人文档」，返回融合重排后的片段。

    个人文档参与与否由 ``personal_docs.rag.enabled`` 与入参共同决定；
    ``user_id`` 为空时退化为纯公共知识库检索（等价于 06 原有行为）。
    """
    cfg = get_personal_docs_config()
    k = int(top_k or cfg["rag"]["top_k"])
    include_personal = bool(
        use_personal_docs and cfg["rag"]["enabled"] and user_id is not None
    )
    return get_knowledge_store().retrieve(
        query,
        top_k=k,
        user_id=user_id,
        use_personal_docs=include_personal,
    )


def rag_context(
    query: str,
    *,
    user_id: Optional[int] = None,
    top_k: Optional[int] = None,
    use_personal_docs: bool = True,
) -> str:
    """把检索结果格式化为带溯源标记的上下文文本（供专家智能体消费）。"""
    return format_passages(
        rag_retrieve(
            query, user_id=user_id, top_k=top_k, use_personal_docs=use_personal_docs
        )
    )
