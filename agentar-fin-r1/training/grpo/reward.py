"""verl reward function for Stage-2 GRPO (paper §3.3, multi-objective reward).

verl calls a single ``compute_score`` per response.  Signature is fixed by
verl's ``RewardManager``::

    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

We expose a graded, multi-objective reward — the paper's "intricate reward
structures":

    reward = w_correct · correctness + w_format · format

* ``correctness`` — verifier-style: 1.0 exact, partial credit by relative
  numeric distance, else 0 (mirrors the §2.3.3 multi-model verifier signal).
* ``format``     — 0.5 for well-formed ``<think>…</think>`` tags, +0.5 for a
  real final answer after them.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# tunable weights (overridable from grpo/config.yaml -> reward.*)
CORRECTNESS_WEIGHT = 1.0
FORMAT_WEIGHT = 0.3

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Answer extraction / grading
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> str | None:
    """Pull the final answer out of a completion (after ``</think>`` / "Answer:")."""
    tail = text.split("</think>", 1)[1] if "</think>" in text else text
    m = re.search(r"(?i)answer\s*[:\-]\s*([^\n]+)", tail)
    if m:
        return m.group(1).strip().rstrip(". ")
    nums = _NUM_RE.findall(tail)
    return nums[-1] if nums else None


def _norm_num(s: str | None) -> float | None:
    if not s:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    return float(m.group()) if m else None


def _matches(pred: str | None, gold: str | None, tol: float = 0.02) -> float:
    """Graded correctness: 1.0 exact, partial credit by rel. numeric distance, else 0."""
    if not pred or not gold:
        return 0.0
    p, g = pred.strip().lower(), gold.strip().lower()
    if p == g:
        return 1.0
    pn, gn = _norm_num(p), _norm_num(g)
    if pn is not None and gn is not None:
        if gn == 0:
            return 1.0 if abs(pn) < 1e-6 else 0.0
        rel = abs(pn - gn) / abs(gn)
        if rel <= 0.25:
            return max(0.0, 1.0 - (rel - tol) / (0.25 - tol))
        return 0.0
    return 1.0 if (g in p or p in g) else 0.0


def format_reward(text: str) -> float:
    """Reward well-structured reasoning: 0.5 for the tags, +0.5 for a real answer."""
    has_think = "<think>" in text and "</think>" in text
    r = 0.5 if has_think else 0.0
    after = text.split("</think>", 1)[1].strip() if has_think else text.strip()
    if after:
        r += 0.5
    return r


# ---------------------------------------------------------------------------
# verl entry point
# ---------------------------------------------------------------------------
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | None,
    extra_info: dict | None = None,
) -> float:
    """verl reward: multi-objective correctness + format (paper §3.3)."""
    correctness = _matches(extract_answer(solution_str), ground_truth) if ground_truth else 0.0
    fmt = format_reward(solution_str)
    return float(CORRECTNESS_WEIGHT * correctness + FORMAT_WEIGHT * fmt)
