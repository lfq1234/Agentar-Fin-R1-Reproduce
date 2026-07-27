"""Synthesis — Track I: Task-Oriented Knowledge-Guided Generation (paper §2.3.2, eq.2/3).

For each task label ℓ ∈ T a dedicated generation agent A_ℓ consumes a knowledge snippet
k ∈ K and emits a reasoning triplet (q, thinking, answer) (eq.2).  Aggregated per label and
union-ed with Track II into ``D_synthesis`` (eq.3, eq.6).

    (q, thinking, answer) = A_ℓ(k ; θ)        eq.(2)
    D_task^ℓ = { (q_i, t_i, a_i) }_{i=1}^{N_ℓ}   eq.(3)

The label→agent mapping is driven by :mod:`finr1_data.labels` (Scene×Task).  This module is
LLM-backed and dry-run safe.
"""
from __future__ import annotations

import logging
from typing import Iterable

from finr1_data.labels import Label, all_labels
from finr1_data.llm import LLMBackend, chat
from finr1_data.schema import KnowledgeUnit, ReasoningTriplet

logger = logging.getLogger(__name__)

# Per-label system prompt templates.  Kept short; extend per task as needed.
_LABEL_SYSTEM: dict[str, str] = {
    "NER": "You are a financial NER specialist. Given a knowledge snippet, write a query "
           "that asks to extract financial entities, show your extraction reasoning, then list entities.",
    "IntentClassification": "You are a financial intent classifier. Given a snippet, write a query "
           "asking to classify user intent, reason about the cues, then give the intent label.",
    "SlotFilling": "You are a financial slot-filling agent. Given a snippet, write a query that "
           "requires filling structured slots, reason about each slot, then output the slots.",
    "EntityDisambiguation": "You are a financial entity disambiguation expert. Given a snippet, "
           "write a query that disambiguates an entity, reason through candidates, then answer.",
    "ConsultationQA": "You are a financial consultation assistant. Given a snippet, write a "
           "consultation query, reason step by step (thinking), then give a precise answer.",
}

_GEN_USER = (
    "Knowledge snippet:\n\"\"\"\n{k}\n\"\"\"\n\n"
    "Produce a reasoning triplet with exactly three clearly delimited parts:\n"
    "Thinking: <step-by-step reasoning>\nAnswer:  <final answer>\n"
    "The query is implied by the snippet; state it first on its own line as 'Query: ...'."
)


def _parse_triplet(text: str) -> tuple[str, str, str]:
    """Parse a model completion into (query, thinking, answer)."""
    query, thinking, answer = "", "", ""
    lines = text.splitlines()
    buf, cur = [], None
    for ln in lines:
        low = ln.lower()
        if low.startswith("query"):
            cur, query, buf = "q", ln.split(":", 1)[-1].strip(), []
        elif low.startswith("thinking"):
            cur, thinking, buf = "t", ln.split(":", 1)[-1].strip(), []
        elif low.startswith("answer"):
            cur, answer, buf = "a", ln.split(":", 1)[-1].strip(), []
        elif cur == "q":
            query += " " + ln.strip()
        elif cur == "t":
            thinking += " " + ln.strip()
        elif cur == "a":
            answer += " " + ln.strip()
    return query.strip(), thinking.strip(), answer.strip()


def generate_for_label(
    label: Label,
    knowledge: list[KnowledgeUnit],
    backend: LLMBackend,
    *,
    n_per_unit: int = 1,
    language: str = "en",
) -> list[ReasoningTriplet]:
    """Generate Track-I triplets for one *label* over the *knowledge* repository."""
    system = _LABEL_SYSTEM.get(label.task, _LABEL_SYSTEM["ConsultationQA"])
    out: list[ReasoningTriplet] = []
    for k in knowledge:
        for j in range(n_per_unit):
            raw = chat(system, _GEN_USER.format(k=k.text), backend)
            q, t, a = _parse_triplet(raw)
            if not q or not a:
                continue
            out.append(ReasoningTriplet(
                id=f"t1-{label.scene}-{label.task}-{len(out):05d}",
                query=q, thinking=t, answer=a,
                scene=label.scene, task=label.task,
                knowledge_id=k.id, track="task", language=language,
            ))
    return out


def track_i_task_oriented(
    knowledge: list[KnowledgeUnit],
    backend: LLMBackend,
    *,
    labels: list[Label] | None = None,
    n_per_unit: int = 1,
    language: str = "en",
) -> list[ReasoningTriplet]:
    """Full Track I (eq.3): generate triplets across all applicable labels, return D_task."""
    labels = labels or all_labels()
    triplets: list[ReasoningTriplet] = []
    for label in labels:
        triplets.extend(generate_for_label(label, knowledge, backend,
                                            n_per_unit=n_per_unit, language=language))
    logger.info("Track I produced %d reasoning triplets across %d labels", len(triplets), len(labels))
    return triplets
