"""08 个人文档与知识图谱：对外响应契约（Pydantic v2，camelCase 输出）。

前端 03（``frontend/src/types/agent.ts``）用 camelCase：
``PersonalDocument{id,filename,size,status,error,uploadedAt,summary}``、
``GraphNode{id,label,type,sourceDocId,properties}``、
``GraphEdge{id,source,target,label,sourceDocId}``。

这里用 ``alias_generator=to_camel`` + ``populate_by_name=True``：
服务层用 snake_case 构造，序列化按 alias 输出 camelCase（路由 ``response_model``
配合 ``response_model_by_alias=True``，FastAPI 默认即为 True）。

RAG 自检端点（``/v1/rag/retrieve``）的模型保持 snake_case，与 ``ChatRequest``
一致——它面向后端联调而非前端 03 面板。
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

DocStatus = Literal["pending", "parsing", "done", "error"]


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class PersonalDocumentOut(_CamelModel):
    """前端 ``PersonalDocument`` 镜像。"""

    id: str
    filename: str
    size: int
    status: DocStatus
    error: Optional[str] = None
    uploaded_at: str
    summary: Optional[str] = None


class GraphNodeOut(_CamelModel):
    """前端 ``GraphNode`` 镜像；``type`` 取 Document/Org/Product/Clause/Metric/Person/Concept。"""

    id: str
    label: str
    type: str
    source_doc_id: str
    properties: Optional[dict[str, Union[str, int]]] = None


class GraphEdgeOut(_CamelModel):
    """前端 ``GraphEdge`` 镜像（source/target 为节点 id）。"""

    id: str
    source: str
    target: str
    label: str
    source_doc_id: str


class KnowledgeGraphOut(_CamelModel):
    """``GET /api/v1/knowledge-graph`` 响应。"""

    nodes: list[GraphNodeOut] = []
    edges: list[GraphEdgeOut] = []


class DocumentListOut(_CamelModel):
    """``POST/GET /api/v1/documents`` 响应（前端读 ``.documents``）。"""

    documents: list[PersonalDocumentOut] = []


# —— RAG 自检端点契约（后端联调用，snake_case） —— #
class RagRetrieveRequest(BaseModel):
    query: str
    user_id: Optional[int] = None
    top_k: int = 3
    use_personal_docs: bool = True


class RagPassageOut(BaseModel):
    content: str
    source: str = ""
    score: float = 0.0
    chapter: str = ""
    doc_id: str = ""
    doc_type: str = ""
    title: str = ""


class RagRetrieveResponse(BaseModel):
    passages: list[RagPassageOut] = []
    context: str = ""
