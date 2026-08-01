"""配置加载模块。

职责：
- 读取 ``backend/config/config.yaml``（YAML）。
- 递归替换 ``${ENV_VAR}`` 占位符（从环境变量取值，未设置则保留原字符串）。
- 加载 ``backend/.env``（若存在）注入环境变量（不覆盖已存在的环境变量）。
- 暴露为 ``config``：嵌套结构，既支持 ``config["model"]["mode"]`` 也支持
  ``config.model.get("mode")``（见技术文档 §3.4 工厂用法）。

字段名门禁（评审问题7）与 ``agentscope`` 相关，见 ``app/model`` 层。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/
_CONFIG_DIR = _BACKEND_ROOT / "config"
_CONFIG_PATH = _CONFIG_DIR / "config.yaml"
_ENV_PATH = _BACKEND_ROOT / ".env"

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


class Config(dict):
    """嵌套字典，同时支持属性访问（``config.model`` 等价于 ``config["model"]``）。

    这样工厂里既可以写 ``config.model.get("mode")``，也可以用标准的
    ``config["model"]["mode"]``。
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 防御性
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def _load_dotenv(path: Path) -> None:
    """读取 ``.env`` 注入环境变量；已存在的变量不覆盖。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def _substitute(obj):
    """递归把字符串中的 ${ENV_VAR} 替换为环境变量值。"""
    if isinstance(obj, dict):
        return {k: _substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v) for v in obj]
    if isinstance(obj, str):

        def _repl(match: "re.Match[str]") -> str:
            return os.environ.get(match.group(1), match.group(0))

        return _ENV_PATTERN.sub(_repl, obj)
    return obj


def _to_config(obj):
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def load_config(path: Path | None = None) -> Config:
    """加载 YAML 配置并替换环境变量占位符。

    参数 ``path`` 用于测试注入自定义配置；默认读 ``backend/config/config.yaml``。
    """
    _load_dotenv(_ENV_PATH)
    cfg_path = path or _CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw = _substitute(raw)
    return _to_config(raw)


config: Config = load_config()
