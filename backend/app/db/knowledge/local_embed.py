"""06/08 进程内嵌入直载（离线可用，06 技术文档 §4 规划的 local_embed 真正落地）。

设计：惰性加载 transformers 模型，对最后一层隐藏状态做 attention-mask 加权平均（mean
pooling），再做 L2 归一化得到句向量。默认模型 ``bert-base-chinese``（HF 缓存已存在，无需联网
下载），可由 ``kb.local_embed.model_path`` 或环境变量 ``LOCAL_EMBED_MODEL_PATH`` 覆盖为任意本地
模型（如 bge-small-zh）。

与 01 ``LocalTransformerModel`` 平行：构造不加载，首次 ``embed`` 才 ``from_pretrained``；
进程内单例，ingest 与 retrieve 复用同一份权重。接口对齐 ``EmbedderInterface.embed``——
``embed(texts) -> list[list[float]]``。
"""
from __future__ import annotations

import os
import threading

_MODEL = None
_TOKENIZER = None
_DEVICE = None
_LOCK = threading.Lock()


def _resolve_model_path() -> str:
    """优先级：env LOCAL_EMBED_MODEL_PATH > config kb.local_embed.model_path > 默认本地 Qwen3-0.6B。

    注：HF 缓存里的 bert-base-chinese 仅有 ref 指针、权重未下载，离线不可用；
    改用本地已完整存在的 Qwen3-0.6B 做 mean-pooling 嵌入（零下载、离线可用）。
    """
    from app import config as config_module

    env = os.environ.get("LOCAL_EMBED_MODEL_PATH")
    if env:
        return env
    kb = config_module.config.get("kb", {}) or {}
    local = kb.get("local_embed") or {}
    path = local.get("model_path") or ""
    return path or r"D:/models/Qwen3-0.6B"


def _load() -> None:
    global _MODEL, _TOKENIZER, _DEVICE
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = _resolve_model_path()
    if os.path.isdir(path):
        # 本地目录：直接加载（允许联网回退）
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModel.from_pretrained(path)
    else:
        # HF repo id：离线优先从缓存加载，避免无网时尝试下载卡死
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mdl = AutoModel.from_pretrained(path, local_files_only=True)
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    _TOKENIZER = tok
    _MODEL = mdl
    _DEVICE = device


def get_local_embedder():
    """返回进程内嵌入单例（首次调用才加载模型）。"""
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                _load()
    return _LocalEmbedder()


class _LocalEmbedder:
    """对齐 app.model.base.EmbedderInterface 的最小子集：``embed(texts)``。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        import torch

        if isinstance(texts, str):
            texts = [texts]
        tok = _TOKENIZER
        mdl = _MODEL
        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
        # last_hidden_state: (B, L, H)；attention_mask: (B, L)
        hs = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (hs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        # L2 归一化：余弦检索前统一归一，距离即内积
        norms = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return norms.cpu().tolist()
