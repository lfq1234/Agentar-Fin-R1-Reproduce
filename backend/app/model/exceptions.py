"""模型调用统一异常（评审问题4 修复）。

两种模式（api / local）下的调用失败统一抛出 ``ModelInvokeError``，
便于上层（未来 ``services/``）做一致的错误处理。触发场景见技术文档 §3.3：
- api_key 缺失 / 占位；
- 网络错误（连接失败 / 5xx）；
- 超时或超出 max_retries 仍失败。
"""


class ModelInvokeError(RuntimeError):
    """模型调用失败的统一异常。"""
