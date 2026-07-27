"""Shared data structures for the Agentar-Fin-R1 data reproduction pipeline.

Mirrors the two papers:

* Main paper §2.3.1 — a *knowledge unit* ``k ∈ K`` is the atomic output of the Source
  (knowledge engineering) stage: a verified, normalized, detoxified, refined fragment of
  financial knowledge.
* Main paper §2.3.2 — the Synthesis stage emits *reasoning triplets*
  ``(q, thinking, answer)`` that, after Verification, become the golden training set.
* DeepFinance-100K paper §2 — each sample also carries multimodal metadata
  (Content / Ability / Complexity / Quality / Language / Task), which we reuse as the
  annotation schema for our synthesized triplets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KnowledgeUnit:
    """A refined knowledge unit ``k`` (paper §2.3.1, eq. after step 4).

    Produced by :mod:`finr1_data.source.knowledge_engineering`.
    """

    id: str
    text: str
    scene: str | None = None          # Label System scene (Banking/Securities/...)
    task: str | None = None           # Label System task (NER/ConsultationQA/...)
    source_doc: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTriplet:
    """A (query, thinking, answer) reasoning triplet (paper §2.3.2, eq.2/3/5/6)."""

    id: str
    query: str
    thinking: str
    answer: str
    # ---- provenance / traceability ----
    scene: str | None = None
    task: str | None = None
    knowledge_id: str | None = None   # which KnowledgeUnit seeded this triplet (Track I)
    track: str = "task"               # "task" (Track I) | "evolution" (Track II)
    seed_query: str | None = None     # original query for self-evolution (Track II)
    # ---- verification / governance ----
    consistency: float | None = None  # multi-model ensemble consistency (eq.7)
    reasoning_valid: bool | None = None
    rating: float | None = None       # rating-model score (eq.10)
    complexity: float | None = None   # 1-10 (DeepFinance-100K metadata)
    quality: float | None = None      # 1-10
    language: str = "en"              # "en" | "zh"
    ability: str | None = None        # Knowledge/Reasoning/Math/Code/...
    content: str | None = None        # domain topic (e.g. Fixed Income)
    passed: bool = False              # survived verify() ∧ clean() ∧ score > tau (eq.11)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReasoningTriplet":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
