"""Difficulty-aware training weights (paper §3.1, Eq.16).

Each training sample's cross-entropy is scaled by its normalised difficulty
weight ``w̃`` so that harder examples dominate the gradient.  Two practical
estimators are provided (a faithful pass@k path can be added later):

* ``complexity`` — use the dataset's native ``Complexity`` (1–10) annotation
  (default; zero generation cost).
* ``heuristic`` — a 6-class task prior (math/analysis weighted higher than
  knowledge-QA), no generation needed.
"""
from __future__ import annotations

from datasets import Dataset


def complexity_difficulty_weights(
    ds: Dataset,
    *,
    floor: float = 0.1,
) -> list[float]:
    """Per-sample weights from the dataset's native ``Complexity`` (1–10) score.

    Harder samples get higher weight (matching the paper's "strategically
    prioritises challenging samples" intent).  Normalised so the mean is 1.0,
    with a lower floor to avoid starving easy samples.
    """
    cxs = ds["complexity"]
    lo, hi = min(cxs), max(cxs)
    span = max(hi - lo, 1e-6)
    w = [floor + (1.0 - floor) * (c - lo) / span for c in cxs]
    mean = sum(w) / len(w)
    return [x / mean for x in w]


def heuristic_difficulty_weights(
    ds: Dataset,
    *,
    priors: dict[str, float] | None = None,
    floor: float = 0.1,
) -> dict[str, float]:
    """Per-label difficulty weights by task type (no generation needed).

    Math / reasoning tasks get higher weight (harder); knowledge QA lower.
    Mirrors the *intent* of the paper's pass@k prioritisation without the
    compute cost.  Normalised so the mean is 1.0.
    """
    defaults = {
        "knowledge_qa": 0.8,
        "nlp": 1.0,
        "text_generation": 0.9,
        "compliance_security": 1.2,
        "math": 1.5,
        "analysis_interpretation": 1.3,
    }
    priors = priors or defaults
    labels = set(ds["task_label"])
    out = {label: priors.get(label, 1.0) for label in labels}
    mean = sum(out.values()) / len(out)
    return {label: max(w / mean, floor) for label, w in out.items()}
