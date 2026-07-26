"""Stage 1 — Financial knowledge injection via large-scale SFT + weighted training."""
from __future__ import annotations


def train_stage1(data_dir: str, output_dir: str) -> None:
    """Train Stage 1: SFT on Fin-R1-300K + general reasoning with difficulty weights.

    Args:
        data_dir:   golden triplets from finr1_data (or a small-prototype subset).
        output_dir: where the Stage-1 checkpoint is saved.

    TODO: load Qwen3-8B, apply LoRA/QLoRA, train using weights from
          finr1_training.weighting (small-prototype scope: single/dual GPU).
    """
    raise NotImplementedError("stage1 SFT not yet implemented")
