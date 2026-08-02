"""LocalEmbedder：本地 vLLM OpenAI 兼容嵌入端点（04 嵌入依赖，01 扩展）。

与 ``LocalModel`` 平行：物理分离的一部分。``embed()`` 同样直连 OpenAI SDK 的
``embeddings.create``，仅 ``base_url`` 指向本地 vLLM 的 ``/v1``（如
``http://localhost:8001/v1``，需该端点加载的是嵌入模型）。

关于 api_key（与 ``LocalModel`` 一致的处理）：本地 vLLM 不校验密钥，配置里的
``EMPTY`` 是预期占位、可透传给 OpenAI 客户端（vLLM 会忽略它），因此直连场景
**不**抛 ``ModelInvokeError``——真正的调用失败（连接失败 / 超时 / 5xx）才抛出。
"""
from __future__ import annotations

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.model.base import EmbedConfig, EmbedderInterface
from app.model.exceptions import ModelInvokeError


class LocalEmbedder(EmbedderInterface):
    """本地模式：vLLM OpenAI 兼容嵌入端点，api_key 通常为占位 EMPTY。"""

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
        """导出 AgentScope openai_embedding 配置（v1 形态）。"""
        return {
            "model_type": "openai_embedding",
            "config_name": self.cfg.model_name,
            "model_name": self.cfg.model_name,
            "api_key": self.cfg.api_key,
            "base_url": self.cfg.base_url,
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._build_client()
            resp = client.embeddings.create(model=self.cfg.model_name, input=texts)
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            raise ModelInvokeError(
                f"LocalEmbedder 调用失败（嵌入端点 {self.cfg.base_url} 是否未启动？）: {exc}"
            ) from exc
        return [item.embedding for item in resp.data]
