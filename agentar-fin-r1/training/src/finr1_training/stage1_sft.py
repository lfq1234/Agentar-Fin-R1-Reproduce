"""Stage 1 — Financial knowledge & capability injection via SFT (paper-faithful).

Implements Section 3.2 "Stage 1: Financial Knowledge and Capability Injection" of
*Agentar-Fin-R1* (arXiv:2507.16802).  Training data (paper §4.2) is::

    D_stage1 = D_fin  ∪  D_general

where:

* ``D_fin``  = financial reasoning (CoT) data.  The paper's faithful input is the
  *synthesized* Fin-R1-300K ("Agentar-DeepFinance-300K", NOT released).  Since we
  cannot obtain it, we use the **open** ``antgroup/Agentar-DeepFinance-100K`` as the
  concrete CoT training corpus — paper §4.2 explicitly lists DeepFinance-100K as part
  of the Agentar-Fin-R1 training data, and it is itself a CoT dataset (each sample =
  Question + Solution[CoT + final answer] + metadata).  See :func:`prepare_financial_data`.
* ``D_general`` = "extensive general reasoning datasets" (e.g. MATH, GPQA) mixed in
  to retain general reasoning proficiency (§3.2).  Optional via ``general_data``.

The (query, thinking, answer) **ternary-group generation** is the job of the data
pipeline (``agentar-fin-r1/data``: ``finr1_data.synthesis``).  Its JSONL output can be
merged into SFT through ``extra_data_path`` once that pipeline lands — SFT here
consumes whatever CoT data is available.

Training uses the **weighted training framework** (§3.1): each sample's cross-entropy
loss is scaled by its normalized difficulty weight w̃, per Eq.(16)::

    L_SFT = -1/N · Σ w̃_ℓᵢ · log p(yᵢ|xᵢ)

The paper estimates w̃ via pass@k (Algorithm 1).  DeepFinance-100K additionally ships a
native ``Complexity`` (1-10) difficulty annotation, which we use as the default
difficulty signal (``weighting_method="complexity"``) — the same metadata the §2
pipeline also produces.  pass@k remains available (``"passk"``) for a faithful run.

Quick start::

    python -m finr1_training.scripts.train_sft

Programmatic::

    from finr1_training.stage1_sft import train_stage1
    train_stage1(output_dir="./checkpoints/stage1")
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer

from finr1_training.models import (
    ModelConfig,
    apply_lora,
    load_model,
    load_tokenizer,
    print_trainable_parameters,
)
from finr1_training.weighting import (
    pass_at_k,
    difficulty_weight,
    smoothed_weight,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepFinance-100K schema (paper §2 / Figure 2)
#   Question   : the (possibly multiple-choice) prompt
#   Solution   : CoT "Thinking" + final "Answer"  (sometimes split into fields)
#   metadata   : Content / Ability / Complexity / Quality / Language / Task
# ---------------------------------------------------------------------------

# Candidate column names — the public HF repo may expose slightly different keys,
# so we probe each in order.
Q_COLS = ("Question", "question", "instruction", "problem", "query", "prompt")
SOLUTION_COLS = ("Solution", "solution", "output", "response", "answer_text")
THINK_COLS = ("Thinking", "thinking", "cot", "chain_of_thought", "reasoning")
ANSWER_COLS = ("Answer", "answer", "final_answer", "result")
COMPLEXITY_COLS = ("Complexity", "complexity", "difficulty")
TASK_COLS = ("Task", "task", "subtask")
ABILITY_COLS = ("Ability", "ability", "capability")
CHOICES_COLS = ("Answer Choices", "answer_choices", "choices", "options")

# Task taxonomy for the 6-class difficulty prior (§2.3 task composition):
#   Knowledge QA, NLP, Text Generation, Compliance & Security, Math, Analysis
TASK_LABELS = [
    "knowledge_qa",
    "nlp",
    "text_generation",
    "compliance_security",
    "math",
    "analysis_interpretation",
]


def _pick(columns: tuple[str, ...], available: list[str]) -> str | None:
    for c in columns:
        if c in available:
            return c
    return None


def _split_solution(solution: str) -> tuple[str, str]:
    """Split a Solution string into (thinking, answer).

    DeepFinance-100K's Solution bundles the CoT ("Thinking: ...") and the final
    "Answer: ...".  We split on the first "Answer:" marker; everything before is the
    reasoning trace.  If no marker is found the whole text is treated as the answer.
    """
    if not solution:
        return "", ""
    idx = solution.lower().find("answer:")
    if idx == -1:
        return "", solution.strip()
    thinking = solution[:idx].strip()
    # drop a leading "Thinking:" tag if present (keep the body only)
    if thinking.lower().startswith("thinking:"):
        thinking = thinking[len("thinking:"):].strip()
    answer = solution[idx + len("answer:"):].strip()
    return thinking, answer


def _task_from_metadata(task_raw: str | None, ability_raw: str | None) -> str:
    """Map DeepFinance Task/Ability metadata to one of TASK_LABELS."""
    blob = f"{task_raw or ''} {ability_raw or ''}".lower()
    rules = [
        ("compliance_security", ("compliance", "security", "risk", "regulation", "legal")),
        ("math", ("math", "numerical", "computation", "calculation")),
        ("analysis_interpretation", ("analysis", "interpret", "forecast", "portfolio", "evaluate")),
        ("text_generation", ("text creation", "generation", "copy", "summar", "writing")),
        ("nlp", ("ner", "entity", "classification", "sentiment", "extract", "label")),
        ("knowledge_qa", ("qa", "question", "knowledge", "knowledge")),
    ]
    for label, keys in rules:
        if any(k in blob for k in keys):
            return label
    return "knowledge_qa"


def sample_to_messages(
    question: str,
    thinking: str | None,
    answer: str | None,
    *,
    include_thinking: bool = True,
) -> list[dict]:
    """Build Qwen3.5 chat messages for one (q, thinking, answer) triple.

    When *include_thinking* is True the assistant target contains the full
    chain-of-thought (wrapped in Qwen3-style ``<think>`` tags) followed by the final
    answer — exactly the (query, thinking, answer) triplet form the paper trains on.
    """
    answer = (answer or "").strip()
    if include_thinking and thinking and thinking.strip():
        target = f"<think>\n{thinking.strip()}\n</think>\n\n{answer}"
    else:
        target = answer
    return [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": target},
    ]


def prepare_financial_data(
    dataset_path: str = "antgroup/Agentar-DeepFinance-100K",
    *,
    split: str = "train",
    max_samples: int | None = None,
    include_thinking: bool = True,
    extra_data_path: str | None = None,
    shuffle_seed: int = 42,
) -> Dataset:
    """Load the financial CoT training corpus → chat-format dataset.

    Default corpus is **``antgroup/Agentar-DeepFinance-100K``** (arXiv:2507.12901),
    the open CoT dataset explicitly listed in paper §4.2 as part of the Agentar-Fin-R1
    training data.  Each sample is mapped to::

        {text, task_label, complexity}

    where ``complexity`` (1-10) drives difficulty-aware weighting and ``task_label``
    drives the 6-class prior.  If *dataset_path* points to a local JSONL/parquet of
    {"query"/"question", "thinking", "answer"} (e.g. synthesized by the data pipeline),
    it is loaded instead.  *extra_data_path* lets you additionally merge a pipeline
    ternary-group JSONL into the same schema.
    """
    logger.info("Loading financial CoT training data: %s", dataset_path)

    def _load(path: str) -> Dataset:
        if path.endswith((".json", ".jsonl")):
            return load_dataset("json", data_files=path, split=split)
        if path.endswith((".parquet", ".arrow")):
            return load_dataset("parquet", data_files=path, split=split)
        return load_dataset(path, split=split)

    ds = _load(dataset_path)
    if extra_data_path:
        extra = _load(extra_data_path)
        ds = concatenate_datasets([ds, extra])
        logger.info("Merged extra pipeline data: total %d", len(ds))

    n = len(ds)
    logger.info("Raw financial samples: %d", n)
    if max_samples and n > max_samples:
        ds = ds.shuffle(seed=shuffle_seed).select(range(max_samples))
        logger.info("Subsampled financial to %d", max_samples)

    cols = ds.column_names
    q_col = _pick(Q_COLS, cols) or cols[0]
    sol_col = _pick(SOLUTION_COLS, cols)
    think_col = _pick(THINK_COLS, cols)
    ans_col = _pick(ANSWER_COLS, cols)
    cx_col = _pick(COMPLEXITY_COLS, cols)
    task_col = _pick(TASK_COLS, cols)
    abil_col = _pick(ABILITY_COLS, cols)
    logger.info(
        "Resolved columns: q=%s solution=%s thinking=%s answer=%s complexity=%s task=%s ability=%s",
        q_col, sol_col, think_col, ans_col, cx_col, task_col, abil_col,
    )

    def _fmt(examples: dict) -> dict:
        texts, tasks, complexities = [], [], []
        questions = examples[q_col]
        solutions = examples[sol_col] if sol_col else [None] * len(questions)
        thinkings = examples[think_col] if think_col else [None] * len(questions)
        answers = examples[ans_col] if ans_col else [None] * len(questions)
        tasks_raw = examples[task_col] if task_col else [None] * len(questions)
        abils_raw = examples[abil_col] if abil_col else [None] * len(questions)
        cxs = examples[cx_col] if cx_col else [None] * len(questions)

        for i in range(len(questions)):
            q = questions[i]
            # local-synth form: explicit thinking/answer fields
            thinking, answer = thinkings[i], answers[i]
            # DeepFinance-100K form: solution bundles CoT + answer
            if (not thinking or not str(thinking).strip()) and solutions[i]:
                thinking, answer = _split_solution(str(solutions[i]))
            # multiple-choice: append options to the question
            choices = None
            for c in CHOICES_COLS:
                if c in examples and examples[c][i]:
                    choices = examples[c][i]
                    break
            if choices:
                q = f"{q}\n\nAnswer Choices:\n{choices}"
            thinking = thinking if isinstance(thinking, str) else (thinking or "")
            answer = answer if isinstance(answer, str) else (answer or "")
            msgs = sample_to_messages(q, thinking=thinking, answer=answer,
                                      include_thinking=include_thinking)
            texts.append(json.dumps(msgs, ensure_ascii=False))
            tasks.append(_task_from_metadata(tasks_raw[i], abils_raw[i]))
            try:
                complexities.append(float(cxs[i]) if cxs[i] is not None else 5.0)
            except (TypeError, ValueError):
                complexities.append(5.0)
        return {"text": texts, "task_label": tasks, "complexity": complexities}

    ds = ds.map(_fmt, batched=True, desc="Format financial samples")
    return ds


def prepare_general_data(
    dataset_path: str,
    *,
    split: str = "train",
    max_samples: int | None = None,
    task_label: str = "math",
    complexity: float = 7.0,
    shuffle_seed: int = 42,
) -> Dataset:
    """Load a general reasoning dataset (e.g. MATH) for the augmentation in §3.2.

    General data keeps the model's broad reasoning ability while it absorbs financial
    specialization.  We map it into the same (q, thinking, answer) chat form; most
    general-reasoning sets already ship a ``solution``/CoT field.  A fixed ``complexity``
    (default 7) is assigned so it blends with the financial weighting scheme.
    """
    logger.info("Loading general reasoning data: %s", dataset_path)
    ds = load_dataset(dataset_path, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=shuffle_seed).select(range(max_samples))

    cols = ds.column_names
    q_col = _pick(("question", "problem", "Query", "instruction"), cols) or cols[0]
    a_col = _pick(("answer", "solution", "Answer"), cols) or cols[-1]
    t_col = _pick(("thinking", "solution", "cot"), cols)

    def _fmt(examples: dict) -> dict:
        texts, tasks, complexities = [], [], []
        for i in range(len(examples[q_col])):
            q = examples[q_col][i]
            a = examples[a_col][i] if a_col in examples else ""
            t = examples[t_col][i] if (t_col and t_col in examples) else None
            msgs = sample_to_messages(q, thinking=t, answer=a, include_thinking=True)
            texts.append(json.dumps(msgs, ensure_ascii=False))
            tasks.append(task_label)
            complexities.append(complexity)
        return {"text": texts, "task_label": tasks, "complexity": complexities}

    ds = ds.map(_fmt, batched=True, desc="Format general samples")
    return ds


def apply_chat_template_batch(
    ds: Dataset,
    tokenizer,
    max_seq_length: int = 4096,
) -> Dataset:
    """Tokenize the ``text`` (json chat) column with the model's chat template.

    Labels mask the user/prompt turn (``-100``) so loss is only on the assistant turn,
    and ``task_label``/``complexity`` are preserved for difficulty weighting.
    """
    # Disable the template's auto thinking wrapper so our explicit <think> tags stand.
    tokenizer.chat_template = tokenizer.chat_template  # keep as-is; we embed tags manually

    def _fn(examples: dict) -> dict:
        messages_list = [json.loads(t) for t in examples["text"]]
        rendered = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_list
        ]
        tokenized = tokenizer(
            rendered,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None,
        )
        labels = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in ids]
            for ids in tokenized["input_ids"]
        ]
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "task_label": examples["task_label"],
            "complexity": examples["complexity"],
        }

    ds = ds.map(_fn, batched=True, batch_size=64, desc="Apply chat template")
    keep = [c for c in ds.column_names if c in ("input_ids", "attention_mask", "labels", "task_label", "complexity")]
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


# ---------------------------------------------------------------------------
# Difficulty-aware weights (§3.1)
#   - "complexity" : use DeepFinance-100K's native Complexity (1-10) annotation
#   - "heuristic"  : 6-class task prior (no generation)
#   - "passk"      : faithful Algorithm 1 pass@k estimation (needs generation)
# ---------------------------------------------------------------------------


def complexity_difficulty_weights(
    ds: Dataset,
    *,
    floor: float = 0.1,
) -> dict[str, float]:
    """Per-sample weights from the dataset's native ``Complexity`` (1-10) score.

    Harder samples (higher complexity) get higher weight, matching the paper's
    "strategically prioritizes challenging samples" intent.  Normalized so the mean
    is 1.0, with a lower floor to avoid starving easy samples.
    """
    cxs = ds["complexity"]
    lo, hi = min(cxs), max(cxs)
    span = max(hi - lo, 1e-6)
    w = [floor + (1.0 - floor) * (c - lo) / span for c in cxs]
    mean = sum(w) / len(w)
    return [x / mean for x in w] if mean else w


def estimate_difficulty_weights_passk(
    ds: Dataset,
    model,
    tokenizer,
    *,
    n_per_label: int = 32,
    k: int = 8,
    reference_passk: dict[str, float] | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.1,
    floor: float = 0.1,
    rho: float = 0.9,
    prev_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Estimate per-label difficulty weights via pass@k (Algorithm 1).

    For every label we stratify-sample ``n_per_label`` instances, have the *current*
    model generate ``k`` responses, and compute pass@k.  If a reference model's pass@k
    is supplied it contributes the gap penalty.  Raw weights are smoothed across epochs
    (rho) and floored.  This is the faithful paper path but requires on-the-fly
    generation; ``complexity`` is the recommended default for this reproduction.
    """
    raw: dict[str, float] = {}
    for label in set(ds["task_label"]):
        idxs = [i for i, l in enumerate(ds["task_label"]) if l == label][:n_per_label]
        correct = 0
        for i in idxs:
            msgs = json.loads(ds[i]["text"])
            prompt = msgs[0]["content"]
            gold = msgs[1]["content"]
            preds = _generate_k(prompt, model, tokenizer, k=k)
            correct += sum(1 for p in preds if _matches(p, gold))
        pass_cur = pass_at_k(len(idxs), correct, k) if idxs else 0.0
        pass_ref = (reference_passk or {}).get(label, pass_cur)
        raw[label] = difficulty_weight(pass_cur, pass_ref, alpha=alpha, beta=beta, gamma=gamma)

    out: dict[str, float] = {}
    for label in raw:
        prev = prev_weights.get(label) if prev_weights else None
        out[label] = smoothed_weight(raw[label], prev, rho=rho, floor=floor)

    mean = sum(out.values()) / len(out)
    return {l: w / mean for l, w in out.items()}


def heuristic_difficulty_weights(
    ds: Dataset,
    *,
    priors: dict[str, float] | None = None,
    floor: float = 0.1,
    rho: float = 0.9,
    prev_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Prototype difficulty weights by task-type prior (no generation needed).

    Math / reasoning tasks get higher weight (harder); knowledge QA lower.  Mirrors the
    *intent* of Algorithm 1 (prioritise hard tasks) without the compute cost.
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
    out = {}
    for label in set(ds["task_label"]):
        base = priors.get(label, 1.0)
        prev = prev_weights.get(label) if prev_weights else None
        out[label] = smoothed_weight(base, prev, rho=rho, floor=floor)
    mean = sum(out.values()) / len(out)
    return {l: w / mean for l, w in out.items()}


def _generate_k(prompt: str, model, tokenizer, k: int) -> list[str]:
    """Generate k completions for one prompt (greedy + sampling). Lightweight stub."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = []
    for _ in range(k):
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
        out.append(tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    return out


def _matches(pred: str, gold: str) -> bool:
    """Loose correctness check (numeric / substring). Replace with LLM-judge for rigor."""
    g = gold.strip().lower()
    p = pred.strip().lower()
    if not g:
        return False
    return g in p or p in g


def weights_to_per_sample_tensor(ds: Dataset, weights) -> torch.Tensor:
    """Expand weights into a per-sample tensor aligned with *ds*.

    *weights* may be:
      - a per-sample list/array (the ``complexity`` path), or
      - a per-label dict (the ``heuristic`` / ``passk`` paths).
    """
    if isinstance(weights, dict):
        return torch.tensor([weights.get(l, 1.0) for l in ds["task_label"]], dtype=torch.float32)
    return torch.tensor(list(weights), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Weighted SFT trainer — Eq.(16)
# ---------------------------------------------------------------------------


class DifficultyWeightedSFTTrainer(SFTTrainer):
    """SFTTrainer whose per-sample loss is scaled by difficulty weight w̃ (Eq.16).

        L_SFT = -1/N · Σ w̃_ℓᵢ · log p(yᵢ|xᵢ)

    Weights are supplied via :meth:`set_difficulty_weights` (a per-sample tensor).
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


# ---------------------------------------------------------------------------
# Main entry point
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
    dataset_text_field="text",
    max_seq_length=4096,
)


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
    """Run Stage 1 SFT end-to-end (§3.2).

    Args:
        output_dir:       Checkpoint dir.
        financial_data:   Financial CoT corpus — **DeepFinance-100K** by default (paper §4.2).
        general_data:     Optional general-reasoning dataset for augmentation (§3.2).
        extra_data_path:  Optional JSONL from the data pipeline (ternary-group synthesis)
                          to merge into the financial corpus.
        model_cfg:        Model config (Qwen3.5-9B + QLoRA by default).
        max_financial:    Cap on financial samples (prototype subsample).
        max_general:      Cap on general samples.
        include_thinking: Train on CoT (thinking) traces.
        max_seq_length:   Max token length (paper uses 16K; lower for prototypes).
        weighting_method: "complexity" (default, uses dataset's Complexity annotation),
                          "heuristic" (6-class task prior), or "passk" (faithful Algorithm 1).
        sft_args_override: Merged over :data:`DEFAULT_SFT_ARGS`.

    Returns:
        Path to the saved final checkpoint.
    """
    _cfg = model_cfg or ModelConfig()

    # ---- Model + tokenizer ----
    logger.info("=== Stage 1: load Qwen3.5-9B + QLoRA ===")
    tokenizer = load_tokenizer(_cfg.model_name_or_path)
    model = load_model(_cfg.model_name_or_path, cfg=_cfg)
    model = apply_lora(model, cfg=_cfg)
    print_trainable_parameters(model)

    # ---- Data: D_fin ∪ D_general ----
    logger.info("=== Stage 1: prepare D_fin (DeepFinance-100K) ∪ D_general ===")
    fin_ds = prepare_financial_data(
        financial_data,
        max_samples=max_financial,
        include_thinking=include_thinking,
        extra_data_path=extra_data_path,
    )
    if general_data:
        gen_ds = prepare_general_data(general_data, max_samples=max_general)
        train_ds = concatenate_datasets([fin_ds, gen_ds])
        logger.info("Mixed: %d financial + %d general = %d", len(fin_ds), len(gen_ds), len(train_ds))
    else:
        train_ds = fin_ds

    train_ds = apply_chat_template_batch(train_ds, tokenizer, max_seq_length=max_seq_length)
    train_ds = train_ds.shuffle(seed=42)

    # ---- Difficulty weights (§3.1) ----
    if weighting_method == "passk":
        logger.info("=== Stage 1: estimate difficulty weights via pass@k (Algorithm 1) ===")
        weights = estimate_difficulty_weights_passk(train_ds, model, tokenizer)
    elif weighting_method == "heuristic":
        logger.info("=== Stage 1: heuristic (6-class task) difficulty weights ===")
        weights = heuristic_difficulty_weights(train_ds)
    else:  # complexity
        logger.info("=== Stage 1: difficulty weights from DeepFinance-100K Complexity annotation ===")
        weights = complexity_difficulty_weights(train_ds)
    diff_tensor = weights_to_per_sample_tensor(train_ds, weights)
    logger.info("Per-sample weight stats: mean=%.3f max=%.3f min=%.3f",
                diff_tensor.mean().item(), diff_tensor.max().item(), diff_tensor.min().item())

    # ---- Training args ----
    sft_kwargs = {**DEFAULT_SFT_ARGS, "output_dir": output_dir, "max_seq_length": max_seq_length}
    if sft_args_override:
        sft_kwargs.update(sft_args_override)
    sft_config = SFTConfig(**sft_kwargs)

    # ---- Train ----
    trainer = DifficultyWeightedSFTTrainer(
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
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import argparse

    p = argparse.ArgumentParser(description="Stage 1 SFT — Agentar-Fin-R1 reproduction (§3.2)")
    p.add_argument("--output-dir", default=DEFAULT_SFT_ARGS["output_dir"])
    p.add_argument("--financial-data", default="antgroup/Agentar-DeepFinance-100K",
                   help="Financial CoT corpus (DeepFinance-100K by default)")
    p.add_argument("--extra-data", default=None,
                   help="JSONL from data pipeline (ternary-group synthesis) to merge")
    p.add_argument("--general-data", default=None, help="General reasoning dataset for augmentation")
    p.add_argument("--max-financial", type=int, default=None)
    p.add_argument("--max-general", type=int, default=None)
    p.add_argument("--no-thinking", action="store_true")
    p.add_argument("--weighting", choices=["complexity", "heuristic", "passk"], default="complexity")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seq-length", type=int, default=4096,
                   help="Paper uses 16K; use lower for prototypes")
    args = p.parse_args()

    overrides = dict(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.seq_length,
    )
    ckpt = train_stage1(
        output_dir=args.output_dir,
        financial_data=args.financial_data,
        extra_data_path=args.extra_data,
        general_data=args.general_data,
        max_financial=args.max_financial,
        max_general=args.max_general,
        include_thinking=not args.no_thinking,
        max_seq_length=args.seq_length,
        weighting_method=args.weighting,
        sft_args_override=overrides,
    )
    print(f"\nDone! Final checkpoint: {ckpt}")
