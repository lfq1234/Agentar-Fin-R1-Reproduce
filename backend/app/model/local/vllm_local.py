"""LocalModel：本地 vLLM OpenAI 兼容端点（技术文档 §3.3 / §2 目标）。

物理分离的一部分（与 ApiModel 各自成类、各自文件夹）。

本期 ``generate()`` 同样直连 OpenAI SDK，仅 base_url 指向本地 vLLM
（如 ``http://localhost:8000/v1``）。

关于 api_key（对评审问题4 "EMPTY 视为缺失" 的工程处理）：
本地 vLLM 不校验密钥，配置里的 ``EMPTY`` 是预期占位、可直接透传给 OpenAI 客户端
（vLLM 会忽略它），因此直连场景**不**抛 ModelInvokeError——真正的调用失败
（连接失败 / 超时 / 5xx）才抛出。若希望严格校验，可在 config.local.api_key 填入任意
非空字符串。此处理与原文档 §3.3 字面措辞不同，目的是让 local 模式真正可运行，
建议同步更新技术文档 §3.3 / §4 步骤5 的说明。
"""
from __future__ import annotations

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.model.base import ModelConfig, ModelInterface
from app.model.exceptions import ModelInvokeError


class LocalModel(ModelInterface):
    """本地模式：vLLM OpenAI 兼容端点，api_key 通常为占位 EMPTY。"""

    def _build_client(self) -> OpenAI:
        # 本地 vLLM 不校验密钥；空串时用占位避免 OpenAI 客户端拒绝构造。
        key = self.cfg.api_key or "not-needed"
        return OpenAI(
            api_key=key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout,
            max_retries=self.cfg.max_retries,
        )

    def build_agentscope_config(self) -> dict:
        return {
            "model_type": "openai_chat",
            "config_name": self.cfg.model_name,
            "model_name": self.cfg.model_name,
            "api_key": self.cfg.api_key,
            "base_url": self.cfg.base_url,
            "temperature": self.cfg.temperature,
            "stream": self.cfg.stream,
        }

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            client = self._build_client()
            resp = client.chat.completions.create(
                model=self.cfg.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.cfg.temperature,
                stream=self.cfg.stream,
                **kwargs,
            )
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            raise ModelInvokeError(
                f"LocalModel 调用失败（vLLM 端点 {self.cfg.base_url} 是否未启动？）: {exc}"
            ) from exc

        if self.cfg.stream:
            parts: list[str] = []
            for chunk in resp:
                for choice in chunk.choices:
                    if choice.delta and choice.delta.content:
                        parts.append(choice.delta.content)
            return "".join(parts)
        return resp.choices[0].message.content or ""
