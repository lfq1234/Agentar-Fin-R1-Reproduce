"""模型接口层测试（技术文档 §5 验收标准 / 评审问题修复）。

覆盖：工厂按 mode 分发、非法 mode 抛 ValueError、build_agentscope_config 产出、
${ENV_VAR} 替换、全局默认合并、异常包装（ModelInvokeError）。

不依赖真实网络：网络失败通过 monkeypatch 客户端触发。
"""
from __future__ import annotations

import pytest

from app import config as config_module
from app.model import ModelInvokeError, ModelInterface, get_model
from app.model.api.openai_api import ApiModel
from app.model.base import ModelConfig
from app.model.local.transformer_local import LocalTransformerModel


def _make_config(mode: str = "api", **model_overrides) -> "config_module.Config":
    base = {
        "model": {
            "mode": mode,
            "timeout": 120,
            "max_retries": 2,
            "api": {
                "model_type": "openai_chat",
                "model_name": "gpt-4o",
                "api_key": "sk-test",
                "base_url": "https://api.openai.com/v1",
                "temperature": 0.3,
                "stream": False,
            },
            "local": {
                "model_type": "openai_chat",
                "model_name": "qwen3-8b-finr1",
                "api_key": "EMPTY",
                "base_url": "http://localhost:8000/v1",
                "temperature": 0.3,
                "stream": False,
            },
        }
    }
    base["model"].update(model_overrides)
    return config_module.Config(base)


@pytest.fixture
def patch_config(monkeypatch):
    """把指定配置注入 app.config.config，供工厂读取。"""
    created = {}

    def _set(mode: str = "api", **model_overrides):
        cfg = _make_config(mode, **model_overrides)
        monkeypatch.setattr(config_module, "config", cfg)
        created["cfg"] = cfg
        return cfg

    _set.cfg = created
    return _set


def test_factory_returns_api(patch_config) -> None:
    patch_config("api")
    m = get_model()
    assert isinstance(m, ApiModel)
    assert isinstance(m, ModelInterface)


def test_factory_returns_local(patch_config) -> None:
    patch_config("local")
    m = get_model()
    assert isinstance(m, LocalTransformerModel)


def test_factory_invalid_mode(patch_config) -> None:  # 评审问题5
    patch_config("local2")
    with pytest.raises(ValueError):
        get_model()


def test_build_agentscope_config_api(patch_config) -> None:  # §5
    patch_config("api")
    d = get_model().build_agentscope_config()
    assert d["model_type"] == "openai_chat"
    assert d["base_url"] == "https://api.openai.com/v1"
    assert d["model_name"] == "gpt-4o"


def test_build_agentscope_config_local(patch_config) -> None:  # §5
    patch_config("local")
    d = get_model().build_agentscope_config()
    assert d["base_url"] == "http://localhost:8000/v1"
    assert d["model_name"] == "qwen3-8b-finr1"


def test_global_default_merged(patch_config) -> None:  # 评审问题1
    patch_config("api")  # 子段无 timeout/max_retries -> 取顶层 120/2
    m = get_model()
    assert m.cfg.timeout == 120
    assert m.cfg.max_retries == 2


def test_subsection_override_global(patch_config) -> None:  # 评审问题1：子段优先
    cfg = patch_config("api", timeout=60)
    cfg["model"]["api"]["timeout"] = 30
    m = get_model()
    assert m.cfg.timeout == 30


def test_apikey_substitution(monkeypatch) -> None:  # §5：环境变量替换
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-123")
    raw = {
        "model": {
            "mode": "api",
            "api": {
                "model_type": "openai_chat",
                "model_name": "gpt-4o",
                "api_key": "${OPENAI_API_KEY}",
                "base_url": "https://api.openai.com/v1",
            },
        }
    }
    # 走真实的 ${ENV_VAR} 替换管线（app/config.load_config 内部调用）
    raw = config_module._substitute(raw)
    cfg = config_module.Config(raw)
    monkeypatch.setattr(config_module, "config", cfg)
    m = get_model()
    assert m.cfg.api_key == "sk-real-123"


def test_api_missing_key_raises(patch_config) -> None:  # 评审问题4
    patch_config(
        "api",
        api={
            "model_type": "openai_chat",
            "model_name": "gpt-4o",
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
        },
    )
    with pytest.raises(ModelInvokeError):
        get_model().generate("hi")


def test_generate_network_error_wrapped(patch_config, monkeypatch) -> None:  # 评审问题4
    patch_config("api")
    m = get_model()

    def _boom(*args, **kwargs):
        from openai import APIConnectionError

        raise APIConnectionError(message="connection refused", request=None)

    monkeypatch.setattr(m, "_build_client", _boom)
    with pytest.raises(ModelInvokeError):
        m.generate("hi")


def test_import_smoke() -> None:  # §5 导入冒烟
    from app.model import get_model, ModelInterface, ModelInvokeError  # noqa: F401

    assert callable(get_model)
    assert issubclass(ModelInterface, object)
