"""模型接口抽象层（技术文档 §3.2）。

- ``ModelConfig``：数据类，承载一次模型调用的全部参数。
- ``ModelInterface``：抽象接口，所有调用实现（api / local）必须实现
  ``build_agentscope_config()`` 与 ``generate()``。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """单个模型的配置。

    工厂（``factory.get_model``）会把配置子段（api / local）合并全局默认后，
    用 ``ModelConfig(**section)`` 构造。
    """

    model_type: str  # 固定 "openai_chat"（评审问题7 字段名以编码前核实为准）
    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.3
    stream: bool = False
    timeout: int = 120
    max_retries: int = 2


class ModelInterface(ABC):
    """统一模型调用抽象接口。

    调用方（未来 ``services/``）只依赖此接口，不关心底层是云端 API 还是本地 vLLM。
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def build_agentscope_config(self) -> dict:
        """返回 AgentScope openai_chat 模型配置字典（供后续 services/ 编排复用）。"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """直连 OpenAI 兼容端点，返回模型文本输出。

        异常：在 key 缺失 / 网络错误 / 超时时抛出 ``ModelInvokeError``（见 exceptions）。
        """
