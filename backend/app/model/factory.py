"""模型工厂（技术文档 §3.4）。

``get_model()`` 读取配置 ``model.mode``，校验合法性后返回对应实现：
- ``api``    -> ``ApiModel``
- ``local``  -> ``LocalModel``

评审问题1 修复：合并顶层全局默认 ``timeout`` / ``max_retries`` 进所选子段（子段优先）。
评审问题5 修复：非法 ``mode`` 显式抛 ``ValueError``，不再静默回退。

通过 ``config_module.config`` 读取配置，便于测试 monkeypatch 注入。
"""
from __future__ import annotations

from app import config as config_module
from app.model.api.openai_api import ApiModel
from app.model.base import ModelConfig, ModelInterface
from app.model.local.vllm_local import LocalModel

_VALID_MODES = ("api", "local")
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

    cls = LocalModel if mode == "local" else ApiModel
    return cls(ModelConfig(**section))
