"""Synthesis stage — dual-track generation + MKE/CoT/SCR (paper §2.3.2, DeepFinance §3)."""
from .task_oriented import track_i_task_oriented
from .self_evolution import track_ii_self_evolution
from .mke import (
    q2a_direct_curation, a2q_counterfactual, t2q_cot_mining,
    sample_cot, scr_rescue,
)

__all__ = [
    "track_i_task_oriented", "track_ii_self_evolution",
    "q2a_direct_curation", "a2q_counterfactual", "t2q_cot_mining",
    "sample_cot", "scr_rescue",
]
