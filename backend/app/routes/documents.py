"""HTTP 路由：08 个人文档与知识图谱（前端 03 双标签面板 + RAG 自检）。

路径（``main.py`` 以 ``prefix="/api"`` 注册，本路由内前缀 ``/v1``，与 chat 一致）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | ``/api/v1/documents`` | 上传（multipart：files[] + user_id） |
| GET | ``/api/v1/documents`` | 列表（倒序） |
| DELETE | ``/api/v1/documents/{doc_id}`` | 级联删除，204 |
| GET | ``/api/v1/documents/{doc_id}/status`` | 单条状态 |
| GET | ``/api/v1/knowledge-graph`` | 图谱 |
| POST | ``/api/v1/rag/retrieve`` | RAG 检索自检（联调用，非前端契约） |

作用域：所有端点强制 ``user_id`` 过滤（沿用 03「不接鉴权、仅 user_id 隔离」约定），
默认值 1 与前端 ``client.ts`` 保持一致。

ingest / 检索均为同步阻塞逻辑，统一经 ``run_in_threadpool`` 执行，避免占住事件循环。
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.db.personal_docs.schemas import (
    DocumentListOut,
    GraphEdgeOut,
    GraphNodeOut,
    KnowledgeGraphOut,
    PersonalDocumentOut,
    RagPassageOut,
    RagRetrieveRequest,
    RagRetrieveResponse,
)
from app.services import documents_service as svc

router = APIRouter(prefix="/v1", tags=["documents"])


def _to_doc_out(row: dict) -> PersonalDocumentOut:
    return PersonalDocumentOut(**row)


@router.post("/documents", response_model=DocumentListOut)
async def upload_documents(
    files: list[UploadFile] = File(...),
    user_id: int = Form(1),
) -> DocumentListOut:
    """上传并同步完成解析 → 切分 → 向量化 → 图谱抽取；单文件失败记 error 不影响其它。"""
    if not files:
        raise HTTPException(status_code=400, detail="未选择文件")
    payload = [(f.filename or "unnamed", await f.read()) for f in files]
    rows = await run_in_threadpool(svc.ingest_files, payload, user_id)
    return DocumentListOut(documents=[_to_doc_out(r) for r in rows])


@router.get("/documents", response_model=DocumentListOut)
async def list_documents(user_id: int = Query(1)) -> DocumentListOut:
    rows = await run_in_threadpool(svc.list_documents, user_id)
    return DocumentListOut(documents=[_to_doc_out(r) for r in rows])


@router.get("/documents/{doc_id}/status", response_model=PersonalDocumentOut)
async def get_document_status(
    doc_id: str, user_id: int = Query(1)
) -> PersonalDocumentOut:
    row = await run_in_threadpool(svc.get_status, doc_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return _to_doc_out(row)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, user_id: int = Query(1)) -> Response:
    ok = await run_in_threadpool(svc.delete_document, doc_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    return Response(status_code=204)


@router.get("/knowledge-graph", response_model=KnowledgeGraphOut)
async def get_knowledge_graph(user_id: int = Query(1)) -> KnowledgeGraphOut:
    nodes, edges = await run_in_threadpool(svc.get_graph, user_id)
    return KnowledgeGraphOut(
        nodes=[GraphNodeOut(**n) for n in nodes],
        edges=[GraphEdgeOut(**e) for e in edges],
    )


@router.post("/rag/retrieve", response_model=RagRetrieveResponse)
async def rag_retrieve(req: RagRetrieveRequest) -> RagRetrieveResponse:
    """RAG 检索自检端点：不经 LLM，直接返回融合重排后的片段与拼好的上下文。

    用于验收「个人文档能被检索到」（评审 AC-R）与前后端联调排障；
    正式问答链路走 ``POST /api/v1/chat``（``use_personal_docs`` 开关）。
    """
    from app.db.knowledge.chunking import format_passages

    passages = await run_in_threadpool(
        lambda: svc.rag_retrieve(
            req.query,
            user_id=req.user_id,
            top_k=req.top_k,
            use_personal_docs=req.use_personal_docs,
        )
    )
    return RagRetrieveResponse(
        passages=[
            RagPassageOut(
                content=p.content,
                source=p.source,
                score=p.score,
                chapter=p.chapter,
                doc_id=p.doc_id,
                doc_type=p.doc_type,
                title=p.title,
            )
            for p in passages
        ],
        context=format_passages(passages),
    )
