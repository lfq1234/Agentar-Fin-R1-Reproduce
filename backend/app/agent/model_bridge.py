"""02-多智能体基础框架：01 模型层的统一调用桥接。

设计边界（评审 G1 · 方案 X）：02 对 LLM 的每一次调用都必须经由 01 的统一入口
`get_model()` 与统一调用方法 `ModelInterface.generate()`。本模块把 AgentScope 的
模型对象契约（`model_type` / `config_name` 属性 + `format()` + `__call__()`）适配到
01 的 `ModelInterface` 上——AgentScope 只负责多智能体编排，模型执行全部落到 01。

实现要点（已用 PoC 验证，对齐 AgentScope 经典版 0.1.6）：
- `format(*args)` 接收 `(memory_list, parser_instruction)`，拍平为单一 prompt 字符串；
- `__call__(prompt)` 调用 `inner.generate(prompt)`，返回 `ModelResponse`；
- `inner.generate()` 抛出的 `ModelInvokeError`（key 缺失 / 网络错误 / 超时）原样上抛，
  由上层 `run()` 捕获转化（评审 N2）。
"""
from __future__ import annotations

from agentscope.message import Msg
from agentscope.models.response import ModelResponse

from app.model import ModelInterface


class AgentarModel:
    """AgentScope 可用的模型对象，但补全实际委托给 01 的 `ModelInterface.generate()`。"""

    # AgentBase 会读取这两个属性（见 agentscope AgentBase 源码），须存在。
    model_type: str = "openai_chat"
    config_name: str = "agentar-bridge"

    def __init__(self, inner: ModelInterface) -> None:
        # inner: 01 返回的 ModelInterface 实现（ApiModel / LocalModel）。
        self._inner = inner

    @staticmethod
    def _msg_to_text(m) -> str:
        role = getattr(m, "name", None) or getattr(m, "role", "user")
        content = getattr(m, "content", m)
        if isinstance(content, list):  # 部分版本 Msg.content 为 block 列表
            content = " ".join(getattr(b, "text", str(b)) for b in content)
        return f"{role}: {content}"

    def format(self, *args) -> str:
        """把 AgentScope 传入的消息拍平成一段 prompt 文本。

        AgentScope 以 `self.model.format(memory, parser_instruction, ...)` 调用，
        这里忽略 parser_instruction，仅把消息逐条拼成可读文本交给 01 的 `generate()`。
        """
        parts: list[str] = []
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, (list, tuple)):
                parts.extend(self._msg_to_text(m) for m in arg)
            elif isinstance(arg, Msg):
                parts.append(self._msg_to_text(arg))
            else:
                parts.append(str(arg))
        return "\n".join(parts)

    def __call__(self, prompt: str, **kwargs) -> ModelResponse:
        """一次补全：委托给 01 的统一调用方法 `generate()`。

        `ModelInvokeError` 由 01 抛出后原样上抛，不在此吞掉（评审 N2）。
        """
        text = self._inner.generate(prompt, **kwargs)
        return ModelResponse(text=text, stream=None)


def extract_content(msg) -> str:
    """从 AgentScope 返回的 `Msg` 中提取纯文本。

    `Msg.content` 在经典版可能是 `str` 或 block 列表，这里统一规整为字符串。
    """
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return " ".join(getattr(b, "text", str(b)) for b in content)
    return content if isinstance(content, str) else str(content)
