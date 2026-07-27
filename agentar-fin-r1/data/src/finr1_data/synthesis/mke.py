"""Synthesis — Multi-perspective Knowledge Extraction (MKE) + CoT sampling + SCR.

Faithful re-implementation of the **DeepFinance-100K** construction pipeline
(dataset paper §3.3–§3.5), which is the *concrete* knowledge-extraction + CoT method we
reuse as the Source/seed generator for our own synthesis.

Three extraction perspectives (§3.3):
  (A) Q2A — Direct Curation: harvest well-structured QA pairs from the seed corpus.
  (B) A2Q — Counterfactual Augmentation: perturb answers (negation / antonym) → LRMs
            generate questions → multi-stage verification.  Bidirectional adversarial.
  (C) T2Q — CoT Knowledge Mining: mine latent knowledge points introduced inside LRM
            CoTs → summarize → build QA pairs.

Then (§3.4) sample multiple CoTs per QA from an LRM and verify answers with a lightweight
verifier (numerical/semantic match).  (§3.5) Self-Corrective Rewriting (SCR) rescues QA
pairs that fail verification: Reflection (diagnose mismatch) → Rewriting (merge reflection
into CoT, regenerate) → re-verify, looping until success or limit.  Final CoT = original CoT
+ alternating (reflection, rewritten) CoTs; final answer = corrected answer.
"""
from __future__ import annotations

import logging
from typing import Iterable

from finr1_data.llm import LLMBackend, Message, answers_match, chat
from finr1_data.schema import ReasoningTriplet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# (A) Q2A — Direct Curation
# ---------------------------------------------------------------------------

def q2a_direct_curation(seed_qa: Iterable[dict], *, drop_low_quality: bool = True) -> list[ReasoningTriplet]:
    """§3.3(A): harvest QA pairs from seed; optional dedup + low-quality filter."""
    seen, out = set(), []
    for rec in seed_qa:
        q = rec.get("question") or rec.get("Question") or rec.get("query") or ""
        a = rec.get("answer") or rec.get("Answer") or ""
        t = rec.get("thinking") or rec.get("Thinking") or rec.get("solution") or ""
        if not q or not a:
            continue
        key = q.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        if drop_low_quality and len(a.strip()) < 3:
            continue
        out.append(ReasoningTriplet(
            id=f"q2a-{len(out):06d}", query=q, thinking=t, answer=a,
            track="task", language=rec.get("language", "en") or "en",
            complexity=_safe_float(rec.get("complexity")),
            quality=_safe_float(rec.get("quality")),
            task=rec.get("task"),
        ))
    logger.info("Q2A curation: %d QA pairs", len(out))
    return out


# ---------------------------------------------------------------------------
# (B) A2Q — Counterfactual Augmentation
# ---------------------------------------------------------------------------

_A2Q_SYSTEM = (
    "You are a counterfactual question generator. Given a (question, answer) pair, perturb "
    "the answer by semantic negation or contextual antonym substitution, then write a NEW "
    "question that fits the perturbed answer. Output ONLY the new question."
)


def a2q_counterfactual(qa_pairs: list[tuple[str, str]], backend: LLMBackend) -> list[ReasoningTriplet]:
    """§3.3(B): adversarial answer perturbation → generate new questions via LRM."""
    out: list[ReasoningTriplet] = []
    for q, a in qa_pairs:
        new_q = chat(_A2Q_SYSTEM, f"Question: {q}\nAnswer: {a}", backend).strip()
        if not new_q:
            continue
        # verification protocol (semantic coherence / logical consistency) — lightweight stand-in
        new_a = chat("Answer the question concisely.", new_q, backend).strip()
        if not new_a:
            continue
        out.append(ReasoningTriplet(
            id=f"a2q-{len(out):06d}", query=new_q, thinking="", answer=new_a,
            track="task", seed_query=q,
        ))
    logger.info("A2Q counterfactual: %d pairs", len(out))
    return out


# ---------------------------------------------------------------------------
# (C) T2Q — CoT Knowledge Mining
# ---------------------------------------------------------------------------

_T2Q_SYSTEM = (
    "Below is a chain-of-thought reasoning trace. Identify the KEY latent knowledge points "
    "it relies on that are NOT common sense, then write 1-2 (question, answer) pairs that "
    "test that knowledge. Output as 'Q: ...\\nA: ...' blocks."
)


def t2q_cot_mining(triplets: list[ReasoningTriplet], backend: LLMBackend) -> list[ReasoningTriplet]:
    """§3.3(C): mine latent knowledge from CoT → new QA pairs."""
    out: list[ReasoningTriplet] = []
    for trip in triplets:
        if not trip.thinking:
            continue
        resp = chat(_T2Q_SYSTEM, trip.thinking, backend)
        for block in resp.split("Q:"):
            block = block.strip()
            if "A:" not in block:
                continue
            q, a = block.split("A:", 1)
            out.append(ReasoningTriplet(
                id=f"t2q-{len(out):06d}", query=q.strip(), thinking="", answer=a.strip(),
                track="task", knowledge_id=trip.id,
            ))
    logger.info("T2Q CoT mining: %d pairs", len(out))
    return out


# ---------------------------------------------------------------------------
# §3.4 CoT Sampling + Verification
# ---------------------------------------------------------------------------

_COT_SYSTEM = (
    "You are a financial reasoning model. Solve the question with explicit step-by-step "
    "chain-of-thought, then give the final answer. End with 'Answer: <answer>'."
)


def sample_cot(qa_pairs: list[tuple[str, str]], backend: LLMBackend, *,
               n_samples: int = 4) -> list[ReasoningTriplet]:
    """§3.4: sample n CoTs per QA from an LRM; keep those whose answer matches the golden."""
    out: list[ReasoningTriplet] = []
    for q, gold in qa_pairs:
        for _ in range(n_samples):
            raw = chat(_COT_SYSTEM, q, backend, temperature=0.8)
            thinking, answer = _split_cot(raw)
            if answers_match(answer, gold):
                out.append(ReasoningTriplet(
                    id=f"cot-{len(out):06d}", query=q, thinking=thinking, answer=answer,
                    track="task",
                ))
    logger.info("CoT sampling: %d verified triplets from %d QA", len(out), len(qa_pairs))
    return out


# ---------------------------------------------------------------------------
# §3.5 Self-Corrective Rewriting (SCR)
# ---------------------------------------------------------------------------

_REFLECT_SYSTEM = (
    "You are a careful teacher. Given a question, a student's INCORRECT answer, and the "
    "GOLD answer, diagnose the student's mistake. Reply with a short 'Reflection: ...' "
    "explaining the error."
)

_REWRITE_SYSTEM = (
    "Continue the reasoning from the original CoT and the reflection below, then give the "
    "correct final answer ending with 'Answer: <answer>'. Reply with 'Thinking: ...' then "
    "'Answer: ...'."
)


def scr_rescue(qa_pairs: list[tuple[str, str]], backend: LLMBackend, *,
               cot_per_q: int = 1, max_iter: int = 3) -> list[ReasoningTriplet]:
    """§3.5: for QA pairs that fail direct CoT sampling, apply SCR to rescue them.

    Loop: sample CoT → if answer matches gold, keep; else Reflection (diagnose) → Rewriting
    (merge reflection into CoT, regenerate) → re-verify.  Final CoT = original + alternating
    (reflection, rewritten); final answer = corrected answer.
    """
    out: list[ReasoningTriplet] = []
    for q, gold in qa_pairs:
        # initial CoT sample
        raw = chat(_COT_SYSTEM, q, backend, temperature=0.8)
        thinking, answer = _split_cot(raw)
        if answers_match(answer, gold):
            out.append(ReasoningTriplet(id=f"scr-{len(out):06d}", query=q,
                                        thinking=thinking, answer=answer, track="task"))
            continue
        # SCR loop
        reflection_acc, rewrite_acc = [], []
        cur_thinking = thinking
        success = False
        for _ in range(max_iter):
            reflection = chat(_REFLECT_SYSTEM,
                              f"Question: {q}\nStudent answer: {answer}\nGold answer: {gold}",
                              backend).strip()
            reflection_acc.append(reflection)
            merged = f"{cur_thinking}\n\nReflection: {reflection}"
            rewritten = chat(_REWRITE_SYSTEM, merged, backend, temperature=0.7)
            r_think, r_answer = _split_cot(rewritten)
            rewrite_acc.append(r_answer)
            cur_thinking = merged + f"\n\nThinking: {r_think}"
            if answers_match(r_answer, gold):
                final_cot = f"{thinking}\n\n" + "\n\n".join(
                    f"Reflection: {reflection_acc[i]}\nRewritten: {rewrite_acc[i]}"
                    for i in range(len(reflection_acc))
                )
                out.append(ReasoningTriplet(id=f"scr-{len(out):06d}", query=q,
                                            thinking=final_cot, answer=r_answer, track="task"))
                success = True
                break
        if not success:
            logger.debug("SCR failed for q=%.60s", q)
    logger.info("SCR rescued %d triplets", len(out))
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _split_cot(text: str) -> tuple[str, str]:
    low = text.lower()
    if "answer:" in low:
        idx = low.find("answer:")
        return text[:idx].replace("thinking:", "").strip(), text[idx + len("answer:"):].strip()
    return text.strip(), ""


def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
