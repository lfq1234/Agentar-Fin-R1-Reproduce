"""CLI entry for Stage-2 GRPO training.

Usage
-----
    # quick prototype on a small hard subset (group_size=4 is the default)
    python -m finr1_training.scripts.train_grpo \
        --hard-subset data/golden/hard_subset.jsonl \
        --output-dir checkpoints/stage2-grpo --max-samples 50

    # from a Stage-1 adapter + custom group size
    python -m finr1_training.scripts.train_grpo \
        --stage1-adapter checkpoints/stage1-sft --group-size 4

Config-driven (configs/grpo_stage2.yaml) values are used as defaults; CLI flags override.
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

from finr1_training.stage2_grpo import GRPOConfig, train_stage2


def _load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-2 GRPO (+ targeted SFT) for Agentar-Fin-R1")
    p.add_argument("--config", default="configs/grpo_stage2.yaml")
    p.add_argument("--hard-subset", default=None, help="JSONL of hard examples (question/answer)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--stage1-adapter", default=None)
    p.add_argument("--group-size", type=int, default=None, help="G rollouts per prompt (default 4)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--beta", type=float, default=None, help="KL coefficient")
    p.add_argument("--temperature", type=float, default=None)
    args = p.parse_args()

    cfg_yaml = _load_yaml(args.config)
    grpo = cfg_yaml.get("grpo", {})
    reward = cfg_yaml.get("reward", {}).get("weights", {})
    tgt = cfg_yaml.get("targeted_sft", {})
    data = cfg_yaml.get("data", {})
    run = cfg_yaml.get("run", {})
    model = cfg_yaml.get("model", {})

    cfg = GRPOConfig(
        group_size=args.group_size or grpo.get("group_size", 4),
        temperature=args.temperature or grpo.get("temperature", 0.9),
        top_p=grpo.get("top_p", 0.95),
        max_new_tokens=grpo.get("max_new_tokens", 1024),
        learning_rate=args.learning_rate or grpo.get("learning_rate", 1e-6),
        beta=args.beta or grpo.get("beta", 0.04),
        clip_eps=grpo.get("clip_eps", 0.2),
        mu=grpo.get("mu", 1),
        epochs=grpo.get("epochs", 1),
        max_grad_norm=grpo.get("max_grad_norm", 1.0),
        reward_weights={
            "correctness": reward.get("correctness", 1.0),
            "format": reward.get("format", 0.3),
            "length": reward.get("length", 0.0),
        },
        stall_patience=tgt.get("stall_patience", 25),
        targeted_sft_steps=tgt.get("targeted_sft_steps", 5),
        log_every=run.get("log_every", 5),
        seed=run.get("seed", 42),
    )

    hard_subset = args.hard_subset or data.get("hard_subset")
    if not hard_subset:
        raise SystemExit("ERROR: provide --hard-subset or set data.hard_subset in config")

    train_stage2(
        hard_subset=hard_subset,
        output_dir=args.output_dir or run.get("output_dir", "checkpoints/stage2-grpo"),
        model_name=args.model_name or model.get("model_name", "Qwen/Qwen3.5-9B"),
        stage1_adapter=args.stage1_adapter or model.get("stage1_adapter"),
        cfg=cfg,
        max_samples=args.max_samples if args.max_samples is not None else data.get("max_samples"),
    )


if __name__ == "__main__":
    main()
