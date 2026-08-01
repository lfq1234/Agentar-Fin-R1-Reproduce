"""模型接口层统一导出（技术文档 §3.1）。

调用方只需 ``from app.model import get_model, ModelInterface, ModelInvokeError``。
"""
from app.model.base import ModelConfig, ModelInterface
from app.model.exceptions import ModelInvokeError
from app.model.factory import get_model

__all__ = ["ModelInterface", "ModelConfig", "get_model", "ModelInvokeError"]
