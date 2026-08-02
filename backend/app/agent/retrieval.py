"""02-多智能体基础框架：RAG 检索接口占位。

本期 RAG 仅做接口占位，真实向量库 / 知识库检索为后续独立需求接入。
流水线结构保持不变：接入真实后端时只替换 `retrieve()` 实现，框架其余代码无需改动。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Passage:
    """一条检索片段。"""

    content: str
    source: str = ""
    score: float = 0.0


def retrieve(query: str, top_k: int = 3) -> list[Passage]:
    """从知识库检索与 `query` 相关的片段。

    Args:
        query (`str`): 检索查询。
        top_k (`int`, defaults to `3`): 返回条数上限。

    Returns:
        `list[Passage]`: 检索结果（本期为占位，恒返回空列表）。
    """
    # TODO(后续需求): 接入向量库 / 知识库 / 监管条文库。
    return []
