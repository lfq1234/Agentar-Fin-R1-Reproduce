"""Convert our Stage-1 error / attribution ``hard_subset.jsonl`` into verl's RLHF format.

verl's ``RLHFDataset`` expects a parquet with three logical columns:

* ``prompt``        — a list of chat messages, e.g. ``[{"role": "user", "content": ...}]``.
                      verl applies the model's chat template at load time.
* ``reward_model``  — a dict with ``ground_truth`` (string). Surfaced to the reward
                      function as ``ground_truth``.
* ``data_source``   — a string that selects which reward function to use.

Input lines (one JSON object each)::

    {"question": "...", "answer": "...", "thinking": "...", "task": "..."}
    # or: {"query": "...", "gold": "..."}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_SOURCE = "agentar_fin"  # must match the reward selector (grpo/reward.py)


def convert_to_verl_parquet(
    hard_subset: str | Path,
    out_parquet: str | Path,
    *,
    max_samples: int | None = None,
    data_source: str = DATA_SOURCE,
) -> str:
    """Read *hard_subset* JSONL and write a verl-ready parquet to *out_parquet*.

    Returns the output parquet path.
    """
    import pandas as pd

    hard_subset = Path(hard_subset)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with hard_subset.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            q = ex.get("question") or ex.get("query")
            g = ex.get("answer") or ex.get("gold")
            if not q:
                continue
            rows.append(
                {
                    "prompt": [{"role": "user", "content": q}],
                    "reward_model": {"ground_truth": g},
                    "data_source": data_source,
                }
            )
            if max_samples and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"No valid samples found in {hard_subset}")

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    logger.info("Wrote %d verl-format samples -> %s", len(df), out_parquet)
    return str(out_parquet)
