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

    path = _resolve_model_path()
    # 优先复用 01 LocalTransformerModel 已加载的共享权重（同路径只存一份），
    # 避免 LLM 与嵌入器各加载一份 Qwen3-0.6B 导致显存/虚拟内存溢出。
    try:
        from app.model.local.transformer_local import _SHARED_LOADED

        cached = _SHARED_LOADED.get(path)
        if cached is not None:
            _DEVICE, _TOKENIZER, _MODEL = cached
            return
    except Exception:
        cached = None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if os.path.isdir(path):
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    else:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=dtype
        )
    mdl.eval()
    mdl.to(device)
    _TOKENIZER = tok
    _MODEL = mdl
    _DEVICE = device
    # 注册到共享缓存，供 LLM 后续复用。
    try:
        from app.model.local.transformer_local import _SHARED_LOADED

        _SHARED_LOADED[path] = (_DEVICE, _TOKENIZER, _MODEL)
    except Exception:
        pass


def get_local_embedder():
    """返回进程内嵌入单例（首次调用才加载模型）。"""
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                _load()
    return _LocalEmbedder()


class _LocalEmbedder:
    """对齐 app.model.base.EmbedderInterface 的最小子集：``embed(texts)``。"""

    # 每批最多处理的行数，避免 output_hidden_states=True 时全部块一次性 batch
    # 导致显存/内存爆炸（28 层 × batch × 512 × 768 ≈ 15MB/chunk，100 块 = 1.5GB 仅隐藏状态）。
    _EMBED_BATCH = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        import torch

        if isinstance(texts, str):
            texts = [texts]
        tok = _TOKENIZER
        mdl = _MODEL

        all_norms: list[list[float]] = []
        for start in range(0, len(texts), self._EMBED_BATCH):
            batch = texts[start : start + self._EMBED_BATCH]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(_DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                out = mdl(**enc, output_hidden_states=True)
            # 取最后一层 hidden_states；(B, L, H)
            # pooling 统一在 float32 做以避免 float16 与 mask dtype 不匹配
            hs = out.hidden_states[-1].float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hs * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            norms = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_norms.extend(norms.cpu().tolist())
        return all_norms
