"""Stage 2 — Hard-task enhancement via GRPO + targeted SFT."""
from __future__ import annotations


def train_stage2(hard_subset: str, output_dir: str) -> None:
    """Train Stage 2: GRPO on difficult subset; fall back to targeted SFT on stalls.

    Args:
        hard_subset: difficult samples (from attribution loop / error analysis).
        output_dir: where the Stage-2 checkpoint is saved.

    TODO: GRPO with multi-objective financial rewards; switch to targeted SFT
          where GRPO fails to converge.
    """
    raise NotImplementedError("stage2 GRPO not yet implemented")
