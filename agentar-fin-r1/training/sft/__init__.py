"""Stage 1 — financial knowledge & capability injection via SFT (paper §3.2).

This is a **drop-in** implementation built entirely on the libraries:

* ``transformers`` + ``peft``  → LoRA adapters on Qwen3.5-9B (see ``model``).
* ``trl.SFTTrainer``           → the supervised-fine-tuning loop.
* A thin :class:`WeightedSFTTrainer` subclass adds the paper's per-sample
  difficulty weighting (Eq.16) on top of the stock trainer — no custom
  optimiser, no hand-rolled loss.

Weighted objective (paper §3.1)::

    L_SFT = -1/N · Σ w̃_ℓᵢ · log p(yᵢ | xᵢ)

Quick start::

    python -m sft

Programmatic::

    from sft import train_stage1
    train_stage1(output_dir="./checkpoints/stage1")
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer

from model import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LORA_TARGET_MODULES,
    DEFAULT_MODEL_NAME,
    ModelConfig,
    apply_lora,
    load_model,
    load_tokenizer,
    print_trainable_parameters,
)
from .data import apply_chat_template_batch, prepare_financial_data, prepare_general_data
from .weighting import complexity_difficulty_weights, heuristic_difficulty_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default training arguments
# ---------------------------------------------------------------------------
DEFAULT_SFT_ARGS = dict(
    output_dir="./checkpoints/stage1",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    eval_strategy="no",
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
    # dataset is pre-tokenized (input_ids / labels); tell SFTTrainer not to
    # look for a raw "text" column.
    dataset_text_field=None,
)


class WeightedSFTTrainer(SFTTrainer):
    """Stock ``SFTTrainer`` whose per-sample loss is scaled by difficulty weight (Eq.16).

    The base trainer already computes the per-token / per-sample cross-entropy;
    we only multiply by the normalised difficulty weight ``w̃`` before averaging.
    Weights are supplied once via :meth:`set_difficulty_weights`.
    """

    def __init__(self, *args: Any, difficulty_weights: torch.Tensor | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._diff_weights: torch.Tensor | None = None
        if difficulty_weights is not None:
            self.set_difficulty_weights(difficulty_weights)

    def set_difficulty_weights(self, weights: torch.Tensor) -> None:
        self._diff_weights = weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if self._diff_weights is not None:
            bs = loss.shape[0] if loss.dim() > 0 else 1
            w = self._diff_weights[:bs].to(loss.device)
            loss = (loss * w).mean()
        return (loss, outputs) if return_outputs else loss


def _build_weights(train_ds, method: str, device) -> torch.Tensor:
    """Resolve the difficulty weights for the chosen *method* into a per-sample tensor."""
    if method == "heuristic":
        logger.info("=== Stage 1: heuristic (6-class task) difficulty weights ===")
        per_label = heuristic_difficulty_weights(train_ds)
        weights = [per_label.get(l, 1.0) for l in train_ds["task_label"]]
    else:  # complexity (default)
        logger.info("=== Stage 1: weights from DeepFinance-100K Complexity annotation ===")
        weights = complexity_difficulty_weights(train_ds)
    t = torch.tensor(weights, dtype=torch.float32, device=device)
    logger.info("Per-sample weight stats: mean=%.3f max=%.3f min=%.3f",
                t.mean().item(), t.max().item(), t.min().item())
    return t


def train_stage1(
    output_dir: str = DEFAULT_SFT_ARGS["output_dir"],
    *,
    financial_data: str = "antgroup/Agentar-DeepFinance-100K",
    general_data: str | None = None,
    extra_data_path: str | None = None,
    model_cfg: ModelConfig | None = None,
    max_financial: int | None = None,
    max_general: int | None = None,
    include_thinking: bool = True,
    max_seq_length: int = 4096,
    weighting_method: str = "complexity",
    sft_args_override: dict | None = None,
) -> str:
    """Run Stage-1 SFT end-to-end (§3.2).

    Args:
        output_dir:        Checkpoint dir.
        financial_data:    Financial CoT corpus — DeepFinance-100K by default (paper §4.2).
        general_data:      Optional general-reasoning dataset for augmentation (§3.2).
        extra_data_path:   Optional JSONL from the data pipeline to merge in.
        model_cfg:         Shared ``ModelConfig`` (Qwen3.5-9B + LoRA).
        max_financial:     Cap on financial samples (prototype subsample).
        max_general:       Cap on general samples.
        include_thinking:  Train on CoT traces.
        max_seq_length:    Max token length (paper uses 16K; lower for prototypes).
        weighting_method:  ``"complexity"`` (default) or ``"heuristic"``.
        sft_args_override: Merged over :data:`DEFAULT_SFT_ARGS`.

    Returns:
        Path to the saved final checkpoint.
    """
    _cfg = model_cfg or ModelConfig()

    # ---- Model + LoRA (peft) ----
    logger.info("=== Stage 1: load %s (precision=%s) + LoRA ===", _cfg.model_name_or_path, _cfg.precision)
    tokenizer = load_tokenizer(_cfg.model_name_or_path)
    model = apply_lora(load_model(_cfg.model_name_or_path, cfg=_cfg), cfg=_cfg)
    print_trainable_parameters(model)

    # ---- Data: D_fin ∪ D_general ----
    logger.info("=== Stage 1: prepare D_fin ∪ D_general ===")
    fin_ds = prepare_financial_data(
        financial_data,
        max_samples=max_financial,
        include_thinking=include_thinking,
        extra_data_path=extra_data_path,
    )
    if general_data:
        from datasets import concatenate_datasets

        gen_ds = prepare_general_data(general_data, max_samples=max_general)
        train_ds = concatenate_datasets([fin_ds, gen_ds])
        logger.info("Mixed: %d financial + %d general = %d", len(fin_ds), len(gen_ds), len(train_ds))
    else:
        train_ds = fin_ds

    train_ds = apply_chat_template_batch(train_ds, tokenizer, max_seq_length=max_seq_length).shuffle(seed=42)

    # ---- Difficulty weights (§3.1) ----
    diff_tensor = _build_weights(train_ds, weighting_method, model.device)

    # ---- Trainer ----
    sft_kwargs = {**DEFAULT_SFT_ARGS, "output_dir": output_dir, "max_seq_length": max_seq_length}
    if sft_args_override:
        sft_kwargs.update(sft_args_override)
    sft_config = SFTConfig(**sft_kwargs)

    trainer = WeightedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        difficulty_weights=diff_tensor,
    )
    logger.info("=== Stage 1: training on %d samples ===", len(train_ds))
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Stage 1 done → %s", final_dir)
    return final_dir


# ---------------------------------------------------------------------------
# CLI  (python -m sft) — YAML defaults from config.yaml, CLI flags override.
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "config.yaml")


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Stage 1 SFT — Agentar-Fin-R1 (peft + trl drop-in, §3.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--financial-data", default=None)
    p.add_argument("--extra-data", default=None)
    p.add_argument("--general-data", default=None)
    p.add_argument("--max-financial", type=int, default=None)
    p.add_argument("--max-general", type=int, default=None)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--weighting", choices=["complexity", "heuristic"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seq-length", type=int, default=None)
    args = p.parse_args()

    cfg = _load_yaml(args.config)
    m, l, d = cfg.get("model", {}), cfg.get("lora", {}), cfg.get("data", {})
    dw, t = cfg.get("difficulty_weighting", {}), cfg.get("training", {})

    model_cfg = ModelConfig(
        model_name_or_path=m.get("name", DEFAULT_MODEL_NAME),
        precision=m.get("precision", "fp16"),
        lora_r=l.get("r", DEFAULT_LORA_R),
        lora_alpha=l.get("alpha", DEFAULT_LORA_ALPHA),
        lora_dropout=l.get("dropout", DEFAULT_LORA_DROPOUT),
        lora_target_modules=l.get("target_modules", list(DEFAULT_LORA_TARGET_MODULES)),
    )

    overrides = dict(
        num_train_epochs=args.epochs or t.get("num_train_epochs", 3),
        per_device_train_batch_size=args.batch_size or t.get("per_device_train_batch_size", 4),
        learning_rate=args.lr or t.get("learning_rate", 2e-4),
        max_seq_length=args.seq_length or d.get("max_seq_length", 4096),
        fp16=t.get("fp16", False),
        bf16=t.get("bf16", False),
    )
    ckpt = train_stage1(
        output_dir=args.output_dir or cfg.get("output_dir", DEFAULT_SFT_ARGS["output_dir"]),
        financial_data=args.financial_data or d.get("financial_data", "antgroup/Agentar-DeepFinance-100K"),
        extra_data_path=args.extra_data or d.get("extra_data"),
        general_data=args.general_data or d.get("general_data"),
        max_financial=args.max_financial or d.get("max_financial"),
        max_general=args.max_general or d.get("max_general"),
        include_thinking=not args.no_thinking if args.no_thinking else d.get("include_thinking", True),
        max_seq_length=args.seq_length or d.get("max_seq_length", 4096),
        weighting_method=args.weighting or dw.get("method", "complexity"),
        sft_args_override=overrides,
        model_cfg=model_cfg,
    )
    print(f"\nDone! Final checkpoint: {ckpt}")
