"""LocalTransformerModel：local 模式下「进程内直接加载权重」的第二种加载方式。

与 ``vllm_local.LocalModel``（连外部 vLLM 端点）并列，二者同属 ``model.mode=local``，
由 ``config.local.loader`` 切换（见 ``factory.get_model``）：

- ``loader: vllm``        -> ``LocalModel``        连外部 OpenAI 兼容端点（默认，零额外依赖）
- ``loader: transformers``-> ``LocalTransformerModel`` 直接在后端进程内用 transformers 加载权重

本类在**后端进程内**用 ``transformers.AutoModelForCausalLM.from_pretrained`` 加载本地
检查点（默认 ``D:/models/Qwen3-0.6B``），不依赖任何外部推理服务进程。它实现与
``LocalModel`` 完全相同的 ``ModelInterface``（``generate`` / ``build_agentscope_config``），
因此上层的 ``services/`` 多智能体编排无需感知底层是端点还是直载。

依赖与运行时（重要）：
- 进程内直载需要 ``torch`` + ``transformers``。为避免污染无 GPU 的轻量运行环境，
  这里**延迟 import**（仅在 ``loader=transformers`` 且实例化时才 import）；若运行时
  未安装则会抛出清晰的错误提示，不影响 ``vllm`` / ``api`` 模式。
- 联调期直载模式请用已含 ``torch+cuda+transformers`` 的 anaconda 运行后端
  （与 ``tools/qwen_server.py`` 同一套环境，已验证可加载并推理）。
"""
from __future__ import annotations

import os

from app.model.base import ModelConfig, ModelInterface
from app.model.exceptions import ModelInvokeError

_DEFAULT_MODEL_PATH = "D:/models/Qwen3-0.6B"


class LocalTransformerModel(ModelInterface):
    """本地模式·进程内直载：transformers 直接加载权重并推理。"""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        model_path = cfg.model_path or os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH).strip()
        self._model_path = model_path
        self._max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "512"))
        self._device, self._tokenizer, self._model = self._load(model_path)

    @staticmethod
    def _load(model_path: str):
        """延迟加载 torch / transformers 并载入本地权重。"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # 运行时缺依赖时给出可操作的提示
            raise ModelInvokeError(
                "LocalTransformerModel 需要 torch 与 transformers，但当前运行环境未安装。"
                "请用已含 torch+cuda+transformers 的解释器运行后端（如 anaconda），"
                "或改用 config.local.loader=vllm 连外部端点。"
            ) from exc

        print(f"[LocalTransformerModel] loading {model_path} ...", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
        except Exception as exc:  # 路径不存在 / 权重损坏等
            raise ModelInvokeError(
                f"LocalTransformerModel 加载权重失败（path={model_path}）: {exc}"
            ) from exc
        print(f"[LocalTransformerModel] loaded device={device}", flush=True)
        return device, tokenizer, model

    def build_agentscope_config(self) -> dict:
        """导出 AgentScope openai_chat 配置（与 LocalModel 同形，供 services/ 编排复用）。

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
