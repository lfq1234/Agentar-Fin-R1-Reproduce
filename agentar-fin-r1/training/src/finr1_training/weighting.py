"""Difficulty-aware training weights via pass@k.

For each task label, sample n instances, generate k responses with the current
model and m reference models, compute pass@k, derive raw weight:

    w_t = alpha*(1 - pass@k_cur) + beta*max(0, pass@k_ref - pass@k_cur) + gamma

Smooth across epochs (rho) with a floor clamp for stability.
"""
from __future__ import annotations

import math


def pass_at_k(n: int, c: int, k: int) -> float:
    """Standard pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def difficulty_weight(
    pass_cur: float,
    pass_ref: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.1,
) -> float:
    """Raw difficulty weight for a task label.

    Higher weight when the current model is weak (low pass@k) and/or lags
    reference models.
    """
    return alpha * (1 - pass_cur) + beta * max(0.0, pass_ref - pass_cur) + gamma


def smoothed_weight(
    raw: float,
    prev: float | None,
    rho: float = 0.9,
    floor: float = 0.1,
) -> float:
    """Exponential smoothing across epochs with a lower-bound clamp."""
    w = raw if prev is None else rho * prev + (1 - rho) * raw
    return max(w, floor)
