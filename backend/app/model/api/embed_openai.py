"""ApiEmbedder：云端 OpenAI 兼容嵌入端点（04 嵌入依赖，01 扩展）。

与 ``ApiModel`` 平行：物理分离的一部分，各自成类、各自文件夹。两者共享抽象接口
``EmbedderInterface`` 与 ``build_agentscope_config`` 字段模板，但实现互不耦合。

``embed()`` 直连 OpenAI SDK 的 ``embeddings.create``（评审问题3 已决策：与
``generate()`` 一致的直连策略）。
"""
from __future__ import annotations

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.model.base import EmbedConfig, EmbedderInterface
from app.model.exceptions import ModelInvokeError

# 视为"无效 key"的占位值：空串、EMPTY、未替换的 ${...} 字面量
_PLACEHOLDER_KEYS = {"", "EMPTY"}


class ApiEmbedder(EmbedderInterface):
    """API 模式：真实 api_key + 云端 base_url 的嵌入模型。"""

    def _build_client(self) -> OpenAI:
        key = self.cfg.api_key
        if key in _PLACEHOLDER_KEYS or key.startswith("${"):
            raise ModelInvokeError(
                f"ApiEmbedder 缺少有效 api_key（当前: {key!r}），"
                "请通过环境变量 OPENAI_API_KEY 注入后再调用。"
            )
        return OpenAI(
            api_key=key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout,
            max_retries=self.cfg.max_retries,
        )

    def build_agentscope_config(self) -> dict:
        """导出 AgentScope openai_embedding 配置（v1 形态，供后续 services/ 编排复用）。

        对齐 AgentScope 经典版（0.1.x）API：`model_type` 为 ``openai_embedding``。
        """
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
            raise ModelInvokeError(f"ApiEmbedder 调用失败: {exc}") from exc
        return [item.embedding for item in resp.data]
