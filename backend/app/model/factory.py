"""模型工厂（技术文档 §3.4 / 嵌入扩展）。

``get_model()`` 读取配置 ``model.mode``，校验合法性后返回对应实现：
- ``api``    -> ``ApiModel``
- ``local``  -> ``LocalTransformerModel``（进程内 transformers 直载，见 transformer_local.py）

``get_embedder()``（01 为 04 扩展）读取配置 ``emb.mode``，校验合法性后返回对应实现：
- ``api``    -> ``ApiEmbedder``
- ``local``  -> ``LocalEmbedder``

评审问题1 修复：合并顶层全局默认 ``timeout`` / ``max_retries`` 进所选子段（子段优先）。
评审问题5 修复：非法 ``mode`` 显式抛 ``ValueError``，不再静默回退。

通过 ``config_module.config`` 读取配置，便于测试 monkeypatch 注入。
"""
from __future__ import annotations

from app import config as config_module
from app.model.api.embed_openai import ApiEmbedder
from app.model.api.openai_api import ApiModel
from app.model.base import EmbedConfig, EmbedderInterface, ModelConfig, ModelInterface
from app.model.local.embed_local import LocalEmbedder
from app.model.local.transformer_local import LocalTransformerModel

_VALID_MODES = ("api", "local")
_VALID_EMB_MODES = ("api", "local")
_GLOBAL_DEFAULT_KEYS = ("timeout", "max_retries")


def get_model() -> ModelInterface:
    model_cfg = config_module.config.get("model", {})
    mode = model_cfg.get("mode")
    if mode not in _VALID_MODES:  # 评审问题5：非法 mode 显式报错
        raise ValueError(f"model.mode 必须是 'api' 或 'local'，收到: {mode!r}")

    # 评审问题1：合并顶层全局默认 timeout/max_retries，子段优先
    section = dict(model_cfg.get(mode, {}))
    for key in _GLOBAL_DEFAULT_KEYS:
        if key not in section and key in model_cfg:
            section[key] = model_cfg[key]

    if mode == "local":
        # local 模式统一走进程内 transformers 直载（外部端点加载方式已移除）。
        return LocalTransformerModel(ModelConfig(**section))
    return ApiModel(ModelConfig(**section))


def get_embedder() -> EmbedderInterface:
    """返回嵌入模型实现（04 的 RAG 向量化调用，见 04 技术文档 §12）。

    读取 ``emb.mode``，校验合法性后分发 ``ApiEmbedder`` / ``LocalEmbedder``，
    与 ``get_model()`` 完全平行（模式校验 + 全局默认合并）。
    """
    emb_cfg = config_module.config.get("emb", {})
    mode = emb_cfg.get("mode")
    if mode not in _VALID_EMB_MODES:
        raise ValueError(f"emb.mode 必须是 'api' 或 'local'，收到: {mode!r}")

    # 合并顶层全局默认 timeout/max_retries，子段优先（与 get_model 一致）
    section = dict(emb_cfg.get(mode, {}))
    for key in _GLOBAL_DEFAULT_KEYS:
        if key not in section and key in emb_cfg:
            section[key] = emb_cfg[key]

    cls = LocalEmbedder if mode == "local" else ApiEmbedder
    return cls(EmbedConfig(**section))
