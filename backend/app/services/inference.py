"""Model inference service.

Loads the checkpoint produced by `agentar-fin-r1/training` and runs generation.
Kept as a stub until the training pipeline emits a model.
"""
from __future__ import annotations

_MODEL: dict | None = None


def load_model(model_path: str | None = None) -> None:
    global _MODEL
    # TODO: load Qwen3-8B + LoRA adapter via transformers / vLLM.
    _MODEL = {"path": model_path, "loaded": False}


def generate(prompt: str, **kwargs) -> str:
    if _MODEL is None:
        load_model()
    # TODO: real generation from the reproduced model.
    return f"[stub generation] {prompt[:80]}"
