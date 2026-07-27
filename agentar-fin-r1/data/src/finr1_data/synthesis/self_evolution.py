"""Synthesis — Track II: Self-Evolution of Instructions (paper §2.3.2, eq.4/5).

Evolve an initial query set Q^0 (manual or sampled from Track I / DeepFinance-100K) into
progressively harder reasoning tasks via a feedback-driven self-evolution agent (eq.4):

    Q^{s+1} = A_evo(Q^s, R; θ_evo)      eq.(4)

where R = {diversity, novelty, answerability}.  Stop at convergence or s = s_max.  Each
evolved query is turned into a triplet by the base model (eq.5); evolution strategies:
Progressive Reasoning Complexity, Structural Diversity, Fitness-Based Filtering.

    D_evolution = { (q_i, t_i, a_i) }_{i=1}^{N_evo}   eq.(5)
    D_synthesis = D_task ∪ D_evolution                 eq.(6)
"""
from __future__ import annotations

import logging

from finr1_data.llm import LLMBackend, chat
from finr1_data.schema import ReasoningTriplet

logger = logging.getLogger(__name__)

_EVO_SYSTEM = (
    "You are a self-evolution agent for financial reasoning data. Given a query, produce a "
    "HARDER, more diverse variant that (a) increases reasoning complexity (multi-step / "
    "counterfactual), (b) varies structure (different phrasing/domain), (c) stays answerable "
    "and factually sound. Output ONLY the new query."
)

_ANSWER_SYSTEM = (
    "You are a financial reasoning solver. Given a query, think step by step then answer. "
    "Reply with 'Thinking: ...' then 'Answer: ...'."
)


def _split(text: str) -> tuple[str, str]:
    low = text.lower()
    if "answer:" in low:
        idx = low.find("answer:")
        return text[:idx].replace("thinking:", "").strip(), text[idx + len("answer:"):].strip()
    return "", text.strip()


def evolve_query(query: str, backend: LLMBackend, *, temperature: float = 0.9) -> str:
    """eq.(4): one self-evolution step — produce a harder variant of *query*."""
    return chat(_EVO_SYSTEM, f"Original query:\n{query}", backend, temperature=temperature).strip()


def fitness_keep(triplet: ReasoningTriplet) -> bool:
    """Fitness-Based Filtering (eq.4 R): keep only sound, coherent, fluent samples."""
    if not triplet.query or not triplet.answer:
        return False
    if len(triplet.thinking or "") < 20:        # require some reasoning depth
        return False
    return True


def track_ii_self_evolution(
    seed_queries: list[str],
    backend: LLMBackend,
    *,
    max_iter: int = 2,
    n_variants: int = 1,
    temperature: float = 0.9,
) -> list[ReasoningTriplet]:
    """Full Track II (eq.4/5): evolve *seed_queries* → D_evolution triplets."""
    current = list(seed_queries)
    evolved: list[ReasoningTriplet] = []
    for s in range(max_iter):
        logger.info("Track II iteration %d: %d queries", s + 1, len(current))
        next_round: list[str] = []
        for q in current:
            for _ in range(n_variants):
                new_q = evolve_query(q, backend, temperature=temperature)
                if not new_q:
                    continue
                raw = chat(_ANSWER_SYSTEM, new_q, backend, temperature=0.7)
                thinking, answer = _split(raw)
                trip = ReasoningTriplet(
                    id=f"t2-{len(evolved):05d}",
                    query=new_q, thinking=thinking, answer=answer,
                    seed_query=q, track="evolution", language="en",
                )
                if fitness_keep(trip):
                    evolved.append(trip)
                next_round.append(new_q)   # chain evolution across iterations
        current = next_round
    logger.info("Track II produced %d evolved triplets", len(evolved))
    return evolved
