"""ApiModel：云端 OpenAI 兼容端点（技术文档 §3.3 / §2 目标）。

物理分离的一部分（与 LocalTransformerModel 各自成类、各自文件夹）。两者共享抽象接口
``ModelInterface`` 与 ``build_agentscope_config`` 字段模板，但实现互不耦合。

本期 ``generate()`` 直连 OpenAI SDK（评审问题3 已决策）。
"""
from __future__ import annotations

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.model.base import ModelConfig, ModelInterface
from app.model.exceptions import ModelInvokeError

# 视为"无效 key"的占位值：空串、EMPTY、未替换的 ${...} 字面量
_PLACEHOLDER_KEYS = {"", "EMPTY"}


class ApiModel(ModelInterface):
    """API 模式：真实 api_key + 云端 base_url。"""

    def _build_client(self) -> OpenAI:
        key = self.cfg.api_key
        if key in _PLACEHOLDER_KEYS or key.startswith("${"):
            raise ModelInvokeError(
                f"ApiModel 缺少有效 api_key（当前: {key!r}），"
                "请通过环境变量 OPENAI_API_KEY 注入后再调用。"
            )
        return OpenAI(
            api_key=key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout,
            max_retries=self.cfg.max_retries,
        )

    def build_agentscope_config(self) -> dict:
        """导出 AgentScope openai_chat 配置（v1 形态，供后续 services/ 编排复用）。

        对齐 AgentScope 经典版（0.1.x）API：`temperature` / `stream` 置于
        `generate_args` 子键（评审 G2 / 01 待修改项已闭环）。
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
            raise ModelInvokeError(f"ApiModel 调用失败: {exc}") from exc

        if self.cfg.stream:
            parts: list[str] = []
            for chunk in resp:
                for choice in chunk.choices:
                    if choice.delta and choice.delta.content:
                        parts.append(choice.delta.content)
            return "".join(parts)
        return resp.choices[0].message.content or ""
