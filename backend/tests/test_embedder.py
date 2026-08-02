"""嵌入接口层测试（01 为 04 扩展：get_embedder / ApiEmbedder / LocalEmbedder）。

覆盖：工厂按 mode 分发、非法 mode 抛 ValueError、build_agentscope_config 产出、
${ENV_VAR} 替换、全局默认合并、异常包装（ModelInvokeError）、真实向量形状（fake client）。

不依赖真实网络：网络失败通过 monkeypatch 客户端触发；向量形状用 fake client 校验。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import config as config_module
from app.model import EmbedderInterface, ModelInvokeError, get_embedder
from app.model.api.embed_openai import ApiEmbedder
from app.model.local.embed_local import LocalEmbedder


def _make_config(mode: str = "api", **overrides) -> "config_module.Config":
    base = {
        "emb": {
            "mode": mode,
            "timeout": 120,
            "max_retries": 2,
            "api": {
                "model_type": "openai_embedding",
                "model_name": "text-embedding-3-small",
                "api_key": "sk-test",
                "base_url": "https://api.openai.com/v1",
            },
            "local": {
                "model_type": "openai_embedding",
                "model_name": "bge-small-zh",
                "api_key": "EMPTY",
                "base_url": "http://localhost:8001/v1",
            },
        }
    }
    base["emb"].update(overrides)
    return config_module.Config(base)


@pytest.fixture
def patch_config(monkeypatch):
    """把指定配置注入 app.config.config，供工厂读取。"""
    created = {}

    def _set(mode: str = "api", **overrides):
        cfg = _make_config(mode, **overrides)
        monkeypatch.setattr(config_module, "config", cfg)
        created["cfg"] = cfg
        return cfg

    _set.cfg = created
    return _set


def test_factory_returns_api(patch_config) -> None:
    patch_config("api")
    e = get_embedder()
    assert isinstance(e, ApiEmbedder)
    assert isinstance(e, EmbedderInterface)


def test_factory_returns_local(patch_config) -> None:
    patch_config("local")
    e = get_embedder()
    assert isinstance(e, LocalEmbedder)


def test_factory_invalid_mode(patch_config) -> None:
    patch_config("local2")
    with pytest.raises(ValueError):
        get_embedder()


def test_build_agentscope_config_api(patch_config) -> None:
    patch_config("api")
    d = get_embedder().build_agentscope_config()
    assert d["model_type"] == "openai_embedding"
    assert d["base_url"] == "https://api.openai.com/v1"
    assert d["model_name"] == "text-embedding-3-small"


def test_build_agentscope_config_local(patch_config) -> None:
    patch_config("local")
    d = get_embedder().build_agentscope_config()
    assert d["base_url"] == "http://localhost:8001/v1"
    assert d["model_name"] == "bge-small-zh"


def test_global_default_merged(patch_config) -> None:  # 与 get_model 一致
    patch_config("api")  # 子段无 timeout/max_retries -> 取顶层 120/2
    e = get_embedder()
    assert e.cfg.timeout == 120
    assert e.cfg.max_retries == 2


def test_subsection_override_global(patch_config) -> None:  # 子段优先
    cfg = patch_config("api", timeout=60)
    cfg["emb"]["api"]["timeout"] = 30
    e = get_embedder()
    assert e.cfg.timeout == 30


def test_apikey_substitution(monkeypatch) -> None:  # 环境变量替换
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-123")
    raw = {
        "emb": {
            "mode": "api",
            "api": {
                "model_type": "openai_embedding",
                "model_name": "text-embedding-3-small",
                "api_key": "${OPENAI_API_KEY}",
                "base_url": "https://api.openai.com/v1",
            },
        }
    }
    raw = config_module._substitute(raw)
    cfg = config_module.Config(raw)
    monkeypatch.setattr(config_module, "config", cfg)
    e = get_embedder()
    assert e.cfg.api_key == "sk-real-123"


def test_api_missing_key_raises(patch_config) -> None:  # 缺 key 抛 ModelInvokeError
    patch_config(
        "api",
        api={
            "model_type": "openai_embedding",
            "model_name": "text-embedding-3-small",
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
        },
    )
    with pytest.raises(ModelInvokeError):
        get_embedder().embed(["hello"])


def test_embed_shape_via_fake_client(patch_config, monkeypatch) -> None:
    patch_config("api")
    e = get_embedder()

    class _FakeResp:
        def __init__(self):
            self.data = [
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
            ]

    class _FakeEmbeddings:
        def create(self, model, input):
            assert isinstance(input, list)
            return _FakeResp()

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    monkeypatch.setattr(e, "_build_client", lambda: _FakeClient())
    vectors = e.embed(["a", "b"])
    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_import_smoke() -> None:  # 导入冒烟
    from app.model import get_embedder, EmbedderInterface, EmbedConfig  # noqa: F401

    assert callable(get_embedder)
    assert issubclass(EmbedderInterface, object)
