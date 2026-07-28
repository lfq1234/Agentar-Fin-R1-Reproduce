"""Stage 2 — hard-task enhancement via GRPO + LoRA (paper §3.3).

This stage is implemented **entirely on the `verl` library** — there is no
hand-rolled GRPO math here.  verl provides the group-relative policy-gradient
trainer, the rollout engine (vLLM / SGLang) and LoRA support out of the box.
This module only:

1. converts the Stage-1 error / attribution ``hard_subset.jsonl`` → a verl
   RLHF parquet (see :mod:`grpo.data`);
2. turns the human-facing ``config.yaml`` into verl hydra overrides;
3. launches ``verl.trainer.main_ppo``.

The GRPO objective (group-relative advantage + KL-to-reference) is verl's
native ``algorithm.adv_estimator=grpo``; the LoRA adapter is enabled via
``actor_rollout_ref.model.lora_rank``.  The multi-objective reward lives in
:mod:`grpo.reward` and is wired in through ``custom_reward_function``.

Quick start::

    python -m grpo --hard-subset data/golden/hard_subset.jsonl --stage1-adapter checkpoints/stage1
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "config.yaml")
_REWARD_PY = str(Path(__file__).resolve().parent / "reward.py")


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _hydra_path(p: str) -> str:
    """Return *p* as a cwd-relative path when possible.

    Hydra override values dislike Windows drive letters (``C:\\...``), so we
    prefer a path relative to the current working directory — ``run_grpo.sh``
    already ``cd``s into ``training/``.  Falls back to the absolute path.
    """
    p = p.replace("\\", "/")
    try:
        rel = Path(p).resolve().relative_to(Path.cwd().resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        return p


def build_verl_overrides(cfg: dict, parquet_path: str, reward_py_path: str) -> list[str]:
    """Translate the human ``config.yaml`` into verl hydra override strings.

    Every key below maps 1:1 onto verl's ``ppo_trainer`` schema, so the launch
    command stays declarative.  ``group_size`` → ``rollout.n`` (the GRPO group);
    ``lora.rank`` → ``model.lora_rank``; the KL coefficient → ``actor.kl_loss_coef``.
    """
    model = cfg.get("model", {})
    lora = cfg.get("lora", {})
    grpo = cfg.get("grpo", {})
    rollout = cfg.get("rollout", {})
    reward = cfg.get("reward", {})
    data = cfg.get("data", {})
    run = cfg.get("run", {})

    n_gpus = int(run.get("n_gpus_per_node", 1))
    train_batch = int(run.get("train_batch_size", 16))
    micro_batch = max(1, train_batch // max(1, n_gpus))

    # reward.py / parquet paths: cwd-relative when possible (hydra-safe on Windows)
    reward_path = _hydra_path(reward_py_path)
    data_files = _hydra_path(parquet_path)

    o = [
        # --- data ---
        f"data.train_files={data_files}",
        f"data.train_batch_size={train_batch}",
        f"data.max_prompt_length={grpo.get('max_prompt_tokens', 2048)}",
        f"data.max_response_length={grpo.get('max_new_tokens', 1024)}",
        # --- model + LoRA (peft, managed by verl) ---
        f"actor_rollout_ref.model.path={model.get('name', 'Qwen/Qwen3.5-9B')}",
        f"actor_rollout_ref.model.lora_rank={int(lora.get('rank', 16))}",
        f"actor_rollout_ref.model.lora_alpha={float(lora.get('alpha', 32))}",
        f"actor_rollout_ref.model.target_modules={lora.get('target_modules', 'all-linear')}",
        "actor_rollout_ref.model.use_shm=True",
    ]
    adapter = model.get("stage1_adapter")
    if adapter:
        o.append(f"actor_rollout_ref.model.lora_adapter_path={adapter}")

    o += [
        # --- actor (policy) optimiser / clipping / KL ---
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={float(grpo.get('beta', 0.04))}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.clip_ratio={float(grpo.get('clip_eps', 0.2))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch}",
        f"actor_rollout_ref.actor.gradient_accumulation_steps={int(grpo.get('mu', 1))}",
        f"actor_rollout_ref.actor.optim.lr={float(grpo.get('learning_rate', 3e-5))}",
        # --- rollout (group size = GRPO `n`) ---
        f"actor_rollout_ref.rollout.name={rollout.get('name', 'vllm')}",
        f"actor_rollout_ref.rollout.n={int(grpo.get('group_size', 8))}",
        f"actor_rollout_ref.rollout.temperature={float(grpo.get('temperature', 0.9))}",
        f"actor_rollout_ref.rollout.top_p={float(grpo.get('top_p', 0.95))}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={float(rollout.get('gpu_memory_utilization', 0.4))}",
        f"actor_rollout_ref.rollout.max_model_len={int(rollout.get('max_model_len', 4096))}",
        f"actor_rollout_ref.rollout.max_num_seqs={int(rollout.get('max_num_seqs', 64))}",
        "actor_rollout_ref.rollout.load_format=safetensors",
        # --- reference policy (KL target) offloaded ---
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        # --- algorithm: GRPO ---
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=True",
        # --- reward ---
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        # --- trainer / logging ---
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.total_epochs={int(run.get('total_epochs', 1))}",
        f"trainer.project_name={run.get('project_name', 'agentar-fin-r1')}",
        f"trainer.experiment_name={run.get('experiment_name', 'stage2-grpo')}",
        f"trainer.default_local_dir={run.get('output_dir', 'checkpoints/stage2-grpo')}",
        "trainer.logger=['console']",
        "trainer.save_freq=10",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
    ]
    return o


def train_stage2(
    hard_subset: str,
    output_dir: str | None = None,
    *,
    config: str | None = None,
    model_name: str | None = None,
    stage1_adapter: str | None = None,
    max_samples: int | None = None,
) -> None:
    """Run Stage-2 GRPO (+ LoRA) on a *hard subset* via verl.

    Args:
        hard_subset:    JSONL of hard examples. Each line needs at least
                        ``question``/``query`` and ``answer``/``gold``.  Typically
                        produced by the attribution loop (``grpo/attribution.py``)
                        or Stage-1 error analysis.
        output_dir:     LoRA adapter output dir (overrides ``run.output_dir``).
        config:         Path to ``config.yaml`` (defaults to this folder's).
        model_name:     Base model id (overrides ``model.name``).
        stage1_adapter: Optional Stage-1 LoRA adapter to warm-start from.
        max_samples:    Cap on hard examples (prototype runs).
    """
    cfg = _load_yaml(config or _DEFAULT_CONFIG)
    if model_name:
        cfg.setdefault("model", {})["name"] = model_name
    if stage1_adapter is not None:
        cfg.setdefault("model", {})["stage1_adapter"] = stage1_adapter
    if output_dir:
        cfg.setdefault("run", {})["output_dir"] = output_dir

    run = cfg.get("run", {})
    data = cfg.get("data", {})

    # 1) hard_subset.jsonl -> verl parquet
    from .data import convert_to_verl_parquet

    parquet_dir = Path(run.get("parquet_dir", "./data/rl"))
    parquet_dir = parquet_dir.resolve() if not parquet_dir.is_absolute() else parquet_dir
    parquet_path = str(parquet_dir / "hard_subset.parquet")
    convert_to_verl_parquet(
        hard_subset,
        parquet_path,
        max_samples=max_samples if max_samples is not None else data.get("max_samples"),
    )

    # 2) config.yaml -> verl overrides
    overrides = build_verl_overrides(cfg, parquet_path, _REWARD_PY)
    logger.info("Launching verl GRPO with %d overrides", len(overrides))

    # 3) launch verl.trainer.main_ppo
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo", *overrides]
    logger.info("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI  (python -m grpo)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage-2 GRPO (+ LoRA) for Agentar-Fin-R1 — verl backend")
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--hard-subset", required=True, help="JSONL of hard examples (question/answer)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--stage1-adapter", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    train_stage2(
        hard_subset=args.hard_subset,
        output_dir=args.output_dir,
        config=args.config,
        model_name=args.model_name,
        stage1_adapter=args.stage1_adapter,
        max_samples=args.max_samples,
    )
