"""模型接口层统一导出（技术文档 §3.1）。

调用方只需 ``from app.model import get_model, get_embedder, ModelInterface,
EmbedderInterface, ModelInvokeError``。
"""
from app.model.base import EmbedConfig, EmbedderInterface, ModelConfig, ModelInterface
from app.model.exceptions import ModelInvokeError
from app.model.factory import get_embedder, get_model

__all__ = [
    "ModelInterface",
    "ModelConfig",
    "get_model",
    "ModelInvokeError",
    "EmbedderInterface",
    "EmbedConfig",
    "get_embedder",
]
