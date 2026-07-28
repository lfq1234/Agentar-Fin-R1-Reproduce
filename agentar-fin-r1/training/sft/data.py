"""Stage-1 data pipeline — DeepFinance-100K → chat-format SFT dataset.

Open HF datasets are messy: column names differ, the chain-of-thought is
bundled inside a single ``Solution`` field, and some samples are
multiple-choice.  This module normalises all of that into a single schema::

    {input_ids, attention_mask, labels, task_label, complexity}

The prompt (user) turn is masked to ``-100`` so the loss is computed only
on the assistant response — the standard practice for causal-LM SFT.
"""
from __future__ import annotations

import json
import logging
from typing import Iterable

from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name probes — open datasets expose slightly different keys, so we
# probe each candidate in order instead of hard-coding one.
# ---------------------------------------------------------------------------
Q_COLS = ("Question", "question", "instruction", "problem", "query", "prompt")
SOLUTION_COLS = ("Solution", "solution", "output", "response", "answer_text")
THINK_COLS = ("Thinking", "thinking", "cot", "chain_of_thought", "reasoning")
ANSWER_COLS = ("Answer", "answer", "final_answer", "result")
COMPLEXITY_COLS = ("Complexity", "complexity", "difficulty")
TASK_COLS = ("Task", "task", "subtask")
ABILITY_COLS = ("Ability", "ability", "capability")
CHOICES_COLS = ("Answer Choices", "answer_choices", "choices", "options")

# 6-class task taxonomy used by the difficulty prior (paper §2.3).
TASK_LABELS = [
    "knowledge_qa",
    "nlp",
    "text_generation",
    "compliance_security",
    "math",
    "analysis_interpretation",
]


def _pick(columns: Iterable[str], available: list[str]) -> str | None:
    for c in columns:
        if c in available:
            return c
    return None


def _split_solution(solution: str) -> tuple[str, str]:
    """Split a ``Solution`` string into (thinking, answer).

    DeepFinance-100K bundles the CoT ("Thinking: …") and the final
    "Answer: …" in one field.  We split on the first "Answer:" marker;
    everything before is the reasoning trace.  No marker → whole text is
    treated as the answer.
    """
    if not solution:
        return "", ""
    idx = solution.lower().find("answer:")
    if idx == -1:
        return "", solution.strip()
    thinking = solution[:idx].strip()
    if thinking.lower().startswith("thinking:"):
        thinking = thinking[len("thinking:"):].strip()
    answer = solution[idx + len("answer:"):].strip()
    return thinking, answer


def _task_from_metadata(task_raw: str | None, ability_raw: str | None) -> str:
    """Map a sample's Task/Ability metadata onto one of :data:`TASK_LABELS`."""
    blob = f"{task_raw or ''} {ability_raw or ''}".lower()
    rules = [
        ("compliance_security", ("compliance", "security", "risk", "regulation", "legal")),
        ("math", ("math", "numerical", "computation", "calculation")),
        ("analysis_interpretation", ("analysis", "interpret", "forecast", "portfolio", "evaluate")),
        ("text_generation", ("text creation", "generation", "copy", "summar", "writing")),
        ("nlp", ("ner", "entity", "classification", "sentiment", "extract", "label")),
        ("knowledge_qa", ("qa", "question", "knowledge")),
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
    """Build Qwen chat messages for one (question, thinking, answer) triple.

    With *include_thinking* the assistant target contains the full CoT
    wrapped in ``<think>`` tags followed by the final answer — exactly the
    (query, thinking, answer) triplet the paper trains on.
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


def _load_any(path: str, split: str) -> Dataset:
    if path.endswith((".json", ".jsonl")):
        return load_dataset("json", data_files=path, split=split)
    if path.endswith((".parquet", ".arrow")):
        return load_dataset("parquet", data_files=path, split=split)
    return load_dataset(path, split=split)


def prepare_financial_data(
    dataset_path: str = "antgroup/Agentar-DeepFinance-100K",
    *,
    split: str = "train",
    max_samples: int | None = None,
    include_thinking: bool = True,
    extra_data_path: str | None = None,
    shuffle_seed: int = 42,
) -> Dataset:
    """Load the financial CoT corpus → dataset with ``{text, task_label, complexity}``.

    Default corpus is **``antgroup/Agentar-DeepFinance-100K``** (paper §4.2).
    If *dataset_path* points to a local JSONL/parquet of
    ``{question, thinking, answer}`` (e.g. synthesised by the data pipeline)
    it is loaded instead.  *extra_data_path* additionally merges a pipeline
    ternary-group JSONL into the same schema.
    """
    logger.info("Loading financial CoT training data: %s", dataset_path)
    ds = _load_any(dataset_path, split)
    if extra_data_path:
        ds = concatenate_datasets([ds, _load_any(extra_data_path, split)])
        logger.info("Merged extra pipeline data: total %d", len(ds))

    if max_samples and len(ds) > max_samples:
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
            thinking, answer = thinkings[i], answers[i]
            if (not thinking or not str(thinking).strip()) and solutions[i]:
                thinking, answer = _split_solution(str(solutions[i]))
            # multiple-choice: append options to the question
            choices = next((examples[c][i] for c in CHOICES_COLS if c in examples and examples[c][i]), None)
            if choices:
                q = f"{q}\n\nAnswer Choices:\n{choices}"
            thinking = thinking if isinstance(thinking, str) else (thinking or "")
            answer = answer if isinstance(answer, str) else (answer or "")
            msgs = sample_to_messages(q, thinking=thinking, answer=answer, include_thinking=include_thinking)
            texts.append(json.dumps(msgs, ensure_ascii=False))
            tasks.append(_task_from_metadata(tasks_raw[i], abils_raw[i]))
            try:
                complexities.append(float(cxs[i]) if cxs[i] is not None else 5.0)
            except (TypeError, ValueError):
                complexities.append(5.0)
        return {"text": texts, "task_label": tasks, "complexity": complexities}

    return ds.map(_fmt, batched=True, desc="Format financial samples")


def prepare_general_data(
    dataset_path: str,
    *,
    split: str = "train",
    max_samples: int | None = None,
    task_label: str = "math",
    complexity: float = 7.0,
    shuffle_seed: int = 42,
) -> Dataset:
    """Load a general-reasoning dataset (e.g. MATH) for the §3.2 augmentation.

    General data keeps broad reasoning ability while the model absorbs
    financial specialisation.  A fixed *complexity* (default 7) blends it
    into the same difficulty-weighting scheme as the financial corpus.
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

    return ds.map(_fmt, batched=True, desc="Format general samples")


def apply_chat_template_batch(
    ds: Dataset,
    tokenizer,
    max_seq_length: int = 4096,
) -> Dataset:
    """Render ``text`` (JSON chat) → tokenised ids with prompt turns masked.

    ``labels`` equals ``input_ids`` everywhere except the user/prompt tokens,
    which are set to ``-100`` so the weighted SFT loss only sees the
    assistant response.  ``task_label`` / ``complexity`` are preserved for
    difficulty weighting.
    """
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
