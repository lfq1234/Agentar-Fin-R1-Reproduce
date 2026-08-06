"""04/08：RAG 检索作用域（把 user_id 透传到 ReAct 工具内部）。

问题：``run()`` 知道 ``user_id``，但检索发生在 ReActAgent 调用的工具函数
（``tools.retrieve_documents``）里，而工具签名由 LLM 按 docstring 生成参数，
不能把 ``user_id`` 暴露成模型可填的入参（模型可能编造别人的 id，越权风险）。

方案：用 ``contextvars`` 在 ``run()`` 内设置一次作用域，工具函数读取即可。
ContextVar 随协程上下文传播，天然按请求隔离，且不改动工具签名 / 提示词。
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class RagScope:
    """一次请求的检索作用域。"""

    user_id: Optional[int] = None
    use_personal_docs: bool = False


_SCOPE: ContextVar[RagScope] = ContextVar("rag_scope", default=RagScope())


def current_scope() -> RagScope:
    return _SCOPE.get()


@contextlib.contextmanager
def scoped(
    user_id: Optional[int] = None, use_personal_docs: bool = False
) -> Iterator[RagScope]:
    """在 with 块内设置检索作用域，退出后自动还原。"""
    scope = RagScope(user_id=user_id, use_personal_docs=use_personal_docs)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)
