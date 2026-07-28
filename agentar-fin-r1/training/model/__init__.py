"""Model loading & QLoRA adapter for Qwen3.5-9B.

Usage::

    from model import load_model, load_tokenizer, apply_lora

    tokenizer = load_tokenizer("Qwen/Qwen3.5-9B")
    model     = load_model("Qwen/Qwen3.5-9B", device_map="auto")
    model     = apply_lora(model)          # returns PeftModel
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config — matches small-prototype scope (single / dual GPU)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-9B"
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class ModelConfig:
    """Centralised hyper-params for model loading + LoRA.

    ``precision`` is the single source of truth for the numeric format — every
    other dtype / quantisation flag is derived from it in ``__post_init__``:

    * ``"int4"``  -> 4-bit NF4 QLoRA (``load_in_4bit=True``), compute dtype bf16.
    * ``"fp16"``  -> full fp16 weights + LoRA, **no quantisation** (default).
    * ``"bf16"``  -> full bf16 weights + LoRA, **no quantisation**.

    Both training stages (SFT, GRPO) reuse this same config so the base model's
    precision is identical across Stage 1 -> Stage 2.
    """

    model_name_or_path: str = DEFAULT_MODEL_NAME
    precision: Literal["int4", "fp16", "bf16"] = "fp16"
    # LoRA
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    lora_target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    trust_remote_code: bool = True

    # --- derived fields (set in __post_init__, do NOT override manually) ---
    load_in_4bit: bool = False
    torch_dtype: torch.dtype = torch.float16
    bnb_compute_dtype: torch.dtype = torch.float16

    def __post_init__(self) -> None:
        if self.precision == "int4":
            self.load_in_4bit = True
            self.torch_dtype = torch.bfloat16
            self.bnb_compute_dtype = torch.bfloat16
        elif self.precision == "bf16":
            self.load_in_4bit = False
            self.torch_dtype = torch.bfloat16
            self.bnb_compute_dtype = torch.bfloat16
        else:  # fp16
            self.load_in_4bit = False
            self.torch_dtype = torch.float16
            self.bnb_compute_dtype = torch.float16


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _bnb_config(cfg: ModelConfig) -> BitsAndBytesConfig | None:
    if not cfg.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=cfg.bnb_compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(
    model_name_or_path: str | None = None,
    *,
    trust_remote_code: bool = True,
    padding_side: str = "right",
) -> AutoTokenizer:
    """Load tokenizer with sensible defaults for chat-style SFT.

    Sets ``padding_side="right"`` so that loss is only computed on the
    assistant tokens (standard practice for causal LM fine-tuning).
    """
    name = model_name_or_path or DEFAULT_MODEL_NAME
    tok = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
    )

    # Qwen3.x uses a special pad token in some variants
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|endoftext|>"})

    tok.padding_side = padding_side
    logger.info("Tokenizer loaded: %s  (pad=%s)", name, tok.pad_token)
    return tok


def load_model(
    model_name_or_path: str | None = None,
    *,
    cfg: ModelConfig | None = None,
    device_map: str | dict = "auto",
) -> AutoModelForCausalLM:
    """Load base model (optionally quantised).

    Returns a raw ``AutoModelForCausalLM`` — call :func:`apply_lora` afterwards
    to wrap it into a ``PeftModel`` ready for training.
    """
    _cfg = cfg or ModelConfig()
    name = model_name_or_path or _cfg.model_name_or_path
    bnb_cfg = _bnb_config(_cfg)

    kw: dict = dict(
        pretrained_model_name_or_path=name,
        torch_dtype=_cfg.torch_dtype,
        device_map=device_map,
        attn_implementation=_cfg.attn_implementation,
        trust_remote_code=_cfg.trust_remote_code,
    )
    if bnb_cfg is not None:
        kw["quantization_config"] = bnb_cfg

    model = AutoModelForCausalLM.from_pretrained(**kw)

    # Print trainable param count
    total = sum(p.numel() for p in model.parameters())
    logger.info("Base model loaded: %s  (%.1fB params)", name, total / 1e9)
    return model


def apply_lora(
    model: AutoModelForCausalLM,
    *,
    cfg: ModelConfig | None = None,
) -> "PeftModelForCausalLM":
    """Wrap *model* with LoRA/QLoRA adapters.

    If the model was loaded in 4-bit this will automatically call
    ``prepare_model_for_kbit_training`` first.
    """
    from peft import PeftModelForCausalLM

    _cfg = cfg or ModelConfig()

    if _cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=_cfg.lora_r,
        lora_alpha=_cfg.lora_alpha,
        target_modules=_cfg.lora_target_modules,
        lora_dropout=_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_cfg)

    # Log trainable fraction
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA applied: r=%d alpha=%d  trainable=%.2fM / %.2fM (%.2f%%)",
        _cfg.lora_r,
        _cfg.lora_alpha,
        trainable / 1e6,
        total / 1e6,
        100 * trainable / total,
    )
    return peft_model


def print_trainable_parameters(model: AutoModelForCausalLM) -> None:
    """Pretty-print trainable vs frozen parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"trainable params: {trainable:,} || "
        f"frozen params: {frozen:,} || "
        f"total: {trainable + frozen:,} || "
        f"% trainable: {100 * trainable / (trainable + frozen):.4f}"
    )
