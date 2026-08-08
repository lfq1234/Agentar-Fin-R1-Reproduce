"""LocalTransformerModel：local 模式下「进程内直接加载权重」的加载方式（唯一）。

``model.mode=local`` 时由 ``factory.get_model`` 直接返回本类（外部端点加载方式已移除）。

本类在**后端进程内**用 ``transformers.AutoModelForCausalLM.from_pretrained`` 加载本地
检查点（默认 ``D:/models/Qwen3-0.6B``），不依赖任何外部推理服务进程。它实现 ``ModelInterface``
（``generate`` / ``build_agentscope_config``），上层 ``services/`` 多智能体编排无需感知底层。

依赖与运行时（重要）：
- 进程内直载需要 ``torch`` + ``transformers``。为避免污染无 GPU 的轻量运行环境，
  这里**延迟 import**（仅在首次 ``generate`` 且实例化后才 import）；若运行时未安装，
  调用时会抛出清晰的错误提示，不影响 ``api`` 模式。
- 联调期请用已含 ``torch+cuda+transformers`` 的解释器运行后端（如 anaconda），
  无需单独启动任何外部推理服务。
"""
from __future__ import annotations

import os
from typing import Any

from app.model.base import ModelConfig, ModelInterface
from app.model.exceptions import ModelInvokeError

_DEFAULT_MODEL_PATH = "D:/models/Qwen3-0.6B"

# 进程内共享缓存：同一路径的本地模型只加载一次，供 LLM 推理与嵌入复用，
# 避免 Qwen3-0.6B 被加载两份导致显存/虚拟内存溢出。
_SHARED_LOADED: dict[str, tuple[str, Any, Any]] = {}


class _ModelLoadError(ModelInvokeError):
    pass


class LocalTransformerModel(ModelInterface):
    """本地模式·进程内直载：transformers 直接加载权重并推理。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._model_path = cfg.model_path or os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH).strip()
        self._max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "512"))
        # 惰性加载：构造时不触碰 torch / 权重，首次 generate 时才真正加载。
        self._device = None
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """首次 generate 时加载 torch / transformers 并载入本地权重（惰性）。"""
        if self._model is not None:
            return
        self._device, self._tokenizer, self._model = self._load(self._model_path)

    @staticmethod
    def _load(model_path: str):
        """延迟加载 torch / transformers 并载入本地权重；同路径只加载一次并共享。"""
        cached = _SHARED_LOADED.get(model_path)
        if cached is not None:
            print(f"[LocalTransformerModel] reuse cached {model_path}", flush=True)
            return cached

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # 运行时缺依赖时给出可操作的提示
            raise ModelInvokeError(
                "LocalTransformerModel 需要 torch 与 transformers，但当前运行环境未安装。"
                "请用已含 torch+cuda+transformers 的解释器运行后端（如 anaconda）。"
            ) from exc

        print(f"[LocalTransformerModel] loading {model_path} ...", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            # 先完整加载到 CPU 内存，再整体迁移到目标设备；避免 device_map="cuda"
            # 触发 accelerate 把权重放到 meta device 而产生 "Cannot copy out of
            # meta tensor" 错误。
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            model.to(device)
        except Exception as exc:  # 路径不存在 / 权重损坏等
            raise ModelInvokeError(
                f"LocalTransformerModel 加载权重失败（path={model_path}）: {exc}"
            ) from exc
        print(f"[LocalTransformerModel] loaded device={device}", flush=True)
        cached = (device, tokenizer, model)
        _SHARED_LOADED[model_path] = cached
        return cached

    def build_agentscope_config(self) -> dict:
        """导出 AgentScope openai_chat 配置（供 services/ 编排复用）。

        注意：实际推理走本类的 ``generate()``（进程内），此字典仅为「配置导出」契约，
        不被真实调用路径使用。
        """
        return {
            "model_type": "openai_chat",
            "config_name": self.cfg.model_name,
            "model_name": self.cfg.model_name,
            "api_key": self.cfg.api_key,
            "base_url": self.cfg.base_url,
            "generate_args": {
                "temperature": self.cfg.temperature,
                "stream": self.cfg.stream,
            },
        }

    def generate(self, prompt: str, **kwargs) -> str:
        self._ensure_loaded()  # 首次调用才真正加载权重
        import torch  # 延迟 import，缺依赖时 generate 也会给出清晰错误

        messages = [{"role": "user", "content": prompt}]
        try:
            text_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3 关闭思考链，直接给答案
            )
        except TypeError:  # 旧版 transformers 无 enable_thinking 参数
            text_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = self._tokenizer(text_prompt, return_tensors="pt").to(self._device)
        do_sample = self.cfg.temperature > 0
        try:
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.cfg.temperature if do_sample else 1.0,
                    top_p=0.9,
                    repetition_penalty=1.05,
                )
        except Exception as exc:  # 推理期 OOM / 显存压力等
            raise ModelInvokeError(
                f"LocalTransformerModel 推理失败（device={self._device}）: {exc}"
            ) from exc

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
