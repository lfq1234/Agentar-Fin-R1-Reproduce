"""Verification & Governance (paper §2.3.3, eq.7–11).

Multi-tier validation before a triplet enters the golden set.

* **Multi-Model Ensemble Verification** (eq.7/8):
    - Consistency (eq.7): M independent models answer the same query; agreement measured by
      a similarity function (lexical + embedding).  consistency = mean pairwise sim.
    - Reasoning validation (eq.8): an independent model checks logical correctness of
      (query, thinking).
* **Human Annotation** (stratified sampling) — placeholder hook; we record sampled ids.
* **Rating Model** (eq.9/10): train/eval a scorer score(q,a) on ensemble∪human signals.
* **Data Governance** (eq.11 cleanser): Deduplication (semantic hash), Detoxification
  (harmful/bias filter), Decontamination (benchmark overlap removal).
* **Final** (eq.11):  D_final = { x ∈ D_synthesis | verify(x) ∧ clean(x) ∧ score(x) > τ }.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable, Sequence

from finr1_data.llm import LLMBackend, Message, chat
from finr1_data.schema import ReasoningTriplet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §2.3.3 Multi-Model Ensemble Verification
# ---------------------------------------------------------------------------

def _lex_sim(a: str, b: str) -> float:
    """Token-overlap Jaccard (lexical part of eq.7 sim)."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _embed_sim(a: str, b: str) -> float:
    """Embedding similarity stand-in. Swap for a real sentence-embedding model.

    Uses a cheap character-ngram cosine so the pipeline runs without torch/sentence-transformers.
    """
    def vec(s: str) -> dict[str, int]:
        s = re.sub(r"\s+", "", s.lower())
        return {s[i:i + 3]: 1 for i in range(len(s) - 2)}
    va, vb = vec(a), vec(b)
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = sum(v * v for v in va.values()) ** 0.5
    nb = sum(v * v for v in vb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def consistency_score(responses: Sequence[str], *, w_lex: float = 0.5) -> float:
    """eq.(7) consistency: mean pairwise similarity (lexical + embedding blend)."""
    if len(responses) < 2:
        return 1.0
    tot, n = 0.0, 0
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            tot += w_lex * _lex_sim(responses[i], responses[j]) + (1 - w_lex) * _embed_sim(responses[i], responses[j])
            n += 1
    return tot / n if n else 1.0


def ensemble_verify(triplet: ReasoningTriplet, backends: Sequence[LLMBackend], *,
                    k: int = 3) -> ReasoningTriplet:
    """Run M models on the query, compute consistency (eq.7) and reasoning validity (eq.8)."""
    responses = [b.complete([Message("user", triplet.query)]) for b in backends[:k]]
    triplet.consistency = consistency_score(responses)
    # eq.8 reasoning validation: independent model judges (query, thinking)
    judge = chat(
        "You are a strict financial verifier. Given a query and its reasoning, reply ONLY "
        "'valid' or 'invalid' with a one-line reason.",
        f"Query: {triplet.query}\nThinking: {triplet.thinking}",
        backends[0],
    ).lower()
    triplet.reasoning_valid = "invalid" not in judge
    return triplet


# ---------------------------------------------------------------------------
# Rating Model (eq.9/10)
# ---------------------------------------------------------------------------

def train_rating_model(signals: Sequence[tuple[ReasoningTriplet, float]]) -> None:
    """eq.(9): rating = ensemble ∪ human.  PLACEHOLDER — fit a real scorer here.

    In production train a classifier/regressor on (embedding(triplet), score) pairs.  We keep
    the interface so downstream code is unchanged; the *application* below uses a heuristic
    proxy until a model is trained.
    """
    logger.info("Rating model training stub: %d signal pairs (fit your scorer here)", len(signals))


def rate_triplet(triplet: ReasoningTriplet) -> float:
    """eq.(10) score(q,a): heuristic proxy = 0.5*consistency + 0.3*valid + 0.2*length_quality.

    Replace with a trained rating model output once :func:`train_rating_model` is wired.
    """
    c = triplet.consistency if triplet.consistency is not None else 0.5
    v = 1.0 if triplet.reasoning_valid else 0.0
    depth = min(1.0, len(triplet.thinking or "") / 400.0)   # reward richer reasoning
    return 0.5 * c + 0.3 * v + 0.2 * depth


# ---------------------------------------------------------------------------
# Data Governance (eq.11 cleanser)
# ---------------------------------------------------------------------------

def _sem_hash(text: str) -> str:
    """Semantic-hash stand-in: normalized 8-char hash for near-dup detection."""
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:8]


def deduplicate(triplets: list[ReasoningTriplet]) -> list[ReasoningTriplet]:
    """Deduplication: drop semantic-hash duplicates, keep the first occurrence."""
    seen, out = set(), []
    for t in triplets:
        h = _sem_hash(t.query + "||" + t.answer)
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    logger.info("Deduplication: %d -> %d", len(triplets), len(out))
    return out


_TOXIC = ("guaranteed profit", "insider tip", "kill", "bomb")


def detoxify(triplets: list[ReasoningTriplet]) -> list[ReasoningTriplet]:
    """Detoxification: drop harmful / biased / non-compliant content."""
    out = [t for t in triplets if not any(term in (t.query + t.answer).lower() for term in _TOXIC)]
    logger.info("Detoxification: %d -> %d", len(triplets), len(out))
    return out


def decontaminate(triplets: list[ReasoningTriplet], benchmark_queries: Iterable[str]) -> list[ReasoningTriplet]:
    """Decontamination: remove samples whose query overlaps an eval benchmark (prevent leakage)."""
    bm = {_sem_hash(q) for q in benchmark_queries}
    out = [t for t in triplets if _sem_hash(t.query) not in bm]
    logger.info("Decontamination: %d -> %d", len(triplets), len(out))
    return out


# ---------------------------------------------------------------------------
# Final composition (eq.11)
# ---------------------------------------------------------------------------

def verify_and_clean(
    triplets: list[ReasoningTriplet],
    backends: Sequence[LLMBackend],
    *,
    tau: float = 0.5,
    benchmark_queries: Iterable[str] = (),
) -> list[ReasoningTriplet]:
    """Apply the full §2.3.3 pipeline: ensemble verify → rate → govern → eq.11 filter."""
    # 1. ensemble verification
    for t in triplets:
        ensemble_verify(t, backends)
        t.rating = rate_triplet(t)
    # 2. governance
    triplets = deduplicate(triplets)
    triplets = detoxify(triplets)
    triplets = decontaminate(triplets, benchmark_queries)
    # 3. eq.11 final gate
    final = [t for t in triplets if (t.rating or 0.0) > tau and (t.reasoning_valid is not False)]
    for t in final:
        t.passed = True
    logger.info("Verification+Governance: %d -> %d passed (tau=%.2f)", len(triplets), len(final), tau)
    return final
