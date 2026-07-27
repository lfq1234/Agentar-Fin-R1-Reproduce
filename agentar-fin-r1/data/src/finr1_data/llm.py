"""LLM backend abstraction for the data pipeline.

The pipeline needs several *generators* / *verifiers* (paper §2.3.2 multi-agent generation,
§2.3.3 multi-model ensemble, DeepFinance-100K §3.4 lightweight verifier + LRM for CoT).
All model access goes through :class:`LLMBackend` so the pipeline is backend-agnostic and
runnable in three modes:

* ``dry-run``  — deterministic template/mock replies; lets the whole pipeline execute
  end-to-end (control-flow, I/O, schema) with **no API key**. Good for tests/prototypes.
* ``openai``   — call an OpenAI-compatible chat endpoint (e.g. a Qwen/QwQ distill server).
* ``hf``       — load a local HuggingFace model via ``transformers``/``vllm`` (TODO).

Swap the backend in one place (``get_backend`` / pipeline config) — nothing else changes.
"""
from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str          # "system" | "user" | "assistant"
    content: str


class LLMBackend(ABC):
    """Abstract text-generation backend."""

    @abstractmethod
    def complete(self, messages: Sequence[Message], **gen_kwargs: Any) -> str:
        """Return the assistant completion for *messages*."""


class DryRunBackend(LLMBackend):
    """Deterministic mock backend — no network, no model.

    Echoes a structured placeholder that keeps the pipeline runnable so you can validate
    wiring, schema and I/O before spending a single token.
    """

    def complete(self, messages: Sequence[Message], **gen_kwargs: Any) -> str:
        last = messages[-1].content if messages else ""
        low = last.lower()
        # Generation prompt asking for a structured (Query/Thinking/Answer) triplet.
        if "reasoning triplet" in low or ("query:" in low and "thinking:" in low and "answer:" in low):
            return (
                "Query: What is the after-tax yield of a 5-year corporate bond priced at 98.5 "
                "with a 6% coupon and a 20% tax rate?\n"
                "Thinking: The after-tax yield adjusts the coupon for taxes. "
                "Pre-tax yield ~ 6 / 98.5 ~ 6.09%. After-tax = 6.09% x (1 - 0.20) ~ 4.87%.\n"
                "Answer: The after-tax yield is approximately 4.87%."
            )
        if "question" in low or "generate a query" in low:
            return (
                "What is the after-tax yield of a 5-year corporate bond priced at 98.5 "
                "with a 6% coupon and a 20% tax rate?"
            )
        if "thinking" in low or "reason" in low:
            return (
                "Thinking: The after-tax yield adjusts the coupon for taxes. "
                "Pre-tax yield ~ 6 / 98.5 ~ 6.09%. After-tax = 6.09% x (1 - 0.20) ~ 4.87%."
            )
        if "answer" in low or "final" in low:
            return "The after-tax yield is approximately 4.87%."
        if "reflection" in low:
            return "The original answer mismatched the golden figure due to a rounding error in the coupon base."
        if "rewrite" in low or "revise" in low:
            return "Rewritten: Using exact coupon base, after-tax yield = 4.87%."
        return "OK."


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible chat backend (works with vLLM / OpenAI / compatible servers)."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 2048) -> None:
        from openai import OpenAI  # lazy import; not required for dry-run
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                              base_url=base_url or os.environ.get("OPENAI_BASE_URL"))

    def complete(self, messages: Sequence[Message], **gen_kwargs: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=gen_kwargs.get("temperature", self.temperature),
            max_tokens=gen_kwargs.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content or ""


def get_backend(mode: str = "dry-run", **kwargs: Any) -> LLMBackend:
    """Factory: build an :class:`LLMBackend` by *mode*."""
    if mode in ("dry-run", "dryrun", "mock"):
        return DryRunBackend()
    if mode in ("openai", "vllm"):
        return OpenAIBackend(model=kwargs.get("model", "Qwen/Qwen3-8B"), base_url=kwargs.get("base_url"))
    if mode == "hf":
        raise NotImplementedError("HF backend TODO: wire transformers/vllm generate() here")
    raise ValueError(f"unknown backend mode: {mode}")


def chat(system: str | None, user: str, backend: LLMBackend, **kw: Any) -> str:
    """One-shot chat call helper."""
    msgs = []
    if system:
        msgs.append(Message("system", system))
    msgs.append(Message("user", user))
    return backend.complete(msgs, **kw)


# ---------------------------------------------------------------------------
# Lightweight answer verifier (DeepFinance-100K §3.4)
# Regex is insufficient for financial answers (e.g. monetary expressions), so a small
# verifier compares normalized numeric values; falls back to substring containment.
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_number(text: str) -> float | None:
    """Extract the first numeric value from *text* (handles 1,234.5 / percentages)."""
    m = _NUM_RE.search(text.replace(",", ""))
    return float(m.group()) if m else None


def answers_match(pred: str, gold: str, tol: float = 1e-3) -> bool:
    """Loose answer match: numeric within tolerance OR substring containment."""
    pn, gn = normalize_number(pred), normalize_number(gold)
    if pn is not None and gn is not None:
        if gn == 0:
            return abs(pn) < 1e-6
        return abs(pn - gn) / abs(gn) <= tol
    p, g = pred.strip().lower(), gold.strip().lower()
    return g in p or p in g
