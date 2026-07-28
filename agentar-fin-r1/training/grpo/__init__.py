"""Stage 2 — Hard-task enhancement via GRPO + Targeted SFT (paper §3.2 / §3.3).

What the paper actually says about Stage 2
------------------------------------------
* "Stage 2: Challenge Task Enhancement … further strengthen the model's performance when
  confronting difficult and challenging problems."  It is a **hybrid** of:
    - GRPO — "Optimizes decision-making capabilities in complex financial scenarios with
      multi-objective considerations and intricate reward structures."
    - Targeted SFT — "Systematically addresses specific performance gaps and weaknesses
      identified through comprehensive Stage 1 evaluation."
* "Tasks demanding sophisticated reasoning (multi-step financial forecasting, comprehensive
  risk assessment, dynamic portfolio optimization) are prioritized."
* "When GRPO encounters convergence challenges on specific task categories, we strategically
  apply targeted SFT using carefully curated high-quality examples."

GRPO itself is the standard group-relative policy-gradient algorithm (Shao et al., 2024 /
DeepSeekMath).  The paper reuses it rather than re-deriving the objective, so we implement
the canonical form:

    For each prompt q sample a *group* of G completions {o_1 … o_G} from the current policy.
    Score each with a reward r_i, then compute the group-relative advantage

        A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)

    and optimise the clipped surrogate plus a KL penalty to a frozen reference policy

        L = - E [ min( ρ_i A_i, clip(ρ_i, 1-ε, 1+ε) A_i ) ] + β · D_KL(π_θ ‖ π_ref)

The user requested **G = 4 rollouts per prompt** (``group_size=4`` below).

The reward is intentionally "intricate / multi-objective" (paper wording):
    reward = w_correct · correctness + w_format · format + w_length · length_penalty
where correctness is verifier-based (numeric / substring match against the golden answer,
mirroring the §2.3.3 multi-model verifier signal) and format rewards well-structured
<think>…</think> reasoning.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from model import ModelConfig, apply_lora, load_model, load_tokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a financial reasoning assistant. Think step by step inside <think>...</think> "
    "tags, then give the final answer after the tags. Be precise with numbers, units and "
    "financial terminology."
)

# ---------------------------------------------------------------------------
# Reward functions (paper: "multi-objective considerations and intricate reward structures")
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """Pull the final answer out of a completion.

    Prefers the text after ``</think>`` and an explicit "Answer: …" marker; falls back to
    the last number in the tail (financial answers are usually numeric).
    """
    tail = text.split("</think>", 1)[1] if "</think>" in text else text
    m = re.search(r"(?i)answer\s*[:\-]\s*([^\n]+)", tail)
    if m:
        return m.group(1).strip().rstrip(". ")
    nums = _NUM_RE.findall(tail)
    return nums[-1] if nums else None


def _norm_num(s: str | None) -> float | None:
    if not s:
        return None
    m = _NUM_RE.search(s.replace(",", ""))
    return float(m.group()) if m else None


def _matches(pred: str | None, gold: str | None, tol: float = 0.02) -> float:
    """Graded correctness: 1.0 exact, partial credit by relative numeric distance, else 0."""
    if not pred or not gold:
        return 0.0
    p, g = pred.strip().lower(), gold.strip().lower()
    if p == g:
        return 1.0
    pn, gn = _norm_num(p), _norm_num(g)
    if pn is not None and gn is not None:
        if gn == 0:
            return 1.0 if abs(pn) < 1e-6 else 0.0
        rel = abs(pn - gn) / abs(gn)
        # within 2% -> full; linearly decays to 0 by 25% relative error
        if rel <= 0.25:
            return max(0.0, 1.0 - (rel - tol) / (0.25 - tol))
        return 0.0
    # string fallback
    return 1.0 if (g in p or p in g) else 0.0


def format_reward(text: str) -> float:
    """Reward well-structured reasoning: 0.5 for the tags, +0.5 for a real final answer."""
    has_think = "<think>" in text and "</think>" in text
    r = 0.5 if has_think else 0.0
    after = text.split("</think>", 1)[1].strip() if has_think else text.strip()
    if after:
        r += 0.5
    return r


def length_reward(text: str, target: int = 512, span: int = 1024) -> float:
    """Mild penalty for overly long completions (keeps rollouts focused)."""
    n = len(text)
    if n <= target:
        return 0.0
    return -min(1.0, (n - target) / span)


def composite_reward(
    text: str,
    gold: str | None,
    weights: dict[str, float],
) -> float:
    """Multi-objective reward (paper §3.2 Stage-2 wording)."""
    r = weights.get("format", 0.3) * format_reward(text)
    if gold is not None:
        r += weights.get("correctness", 1.0) * _matches(extract_answer(text), gold)
    r += weights.get("length", 0.0) * length_reward(text)
    return float(r)


# ---------------------------------------------------------------------------
# Generation + log-probability helpers
# ---------------------------------------------------------------------------
def _build_prompt_ids(tokenizer, question: str, system_prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    mask = torch.ones_like(ids)
    return ids, mask


def _token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    comp_mask: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-prob of the *actual* next token at completion positions.

    Returns (token_logp, comp_shift) both shape (B, L-1), where comp_shift marks the
    completion tokens that we predict (i.e. the token AFTER a completion position).
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logp = out.logits.log_softmax(-1)            # (B, L, V)
    shift_logp = logp[:, :-1, :]                 # (B, L-1, V)
    shift_labels = input_ids[:, 1:]              # (B, L-1)
    token_logp = shift_logp.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    comp_shift = comp_mask[:, 1:].to(token_logp.dtype)  # predict target of comp token
    return token_logp, comp_shift


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    # --- rollout / group ---
    group_size: int = 8                 # G rollouts per prompt (requirement: 8)
    temperature: float = 0.9            # sampling temperature for rollouts
    top_p: float = 0.95
    max_new_tokens: int = 1024
    max_prompt_tokens: int = 2048
    # --- optimisation ---
    learning_rate: float = 1e-6
    beta: float = 0.04                  # KL penalty coefficient (GRPO default ~0.04)
    clip_eps: float = 0.2               # PPO-style clip epsilon
    mu: int = 1                         # inner optimisation epochs per batch
    epochs: int = 1                     # passes over the hard subset
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    # --- reward ---
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {"correctness": 1.0, "format": 0.3, "length": 0.0}
    )
    # --- targeted-SFT fallback (paper: apply when GRPO stalls on a category) ---
    stall_patience: int = 25            # steps w/o reward improvement -> run targeted SFT
    targeted_sft_steps: int = 5
    # --- misc ---
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    grad_accum: int = 1
    log_every: int = 5
    seed: int = 42


# ---------------------------------------------------------------------------
# GRPO trainer
# ---------------------------------------------------------------------------
class GRPOTrainer:
    """Standard GRPO with group size G, group-relative advantage, KL-to-reference."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        cfg: GRPOConfig,
        reward_fn: Callable[[str, str | None], float],
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.device = next(model.parameters()).device

        self.optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
        )
        self._best_reward = -math.inf
        self._steps_since_improve = 0

    # ---- rollout ----
    def _generate_group(self, prompt_ids, prompt_mask):
        G = self.cfg.group_size
        b_ids = prompt_ids.repeat(G, 1)
        b_mask = prompt_mask.repeat(G, 1)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=b_ids,
                attention_mask=b_mask,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = prompt_ids.shape[1]
        comp_ids = gen[:, prompt_len:]                       # (G, C)
        comp_mask = (comp_ids != self.tokenizer.pad_token_id).long()
        texts = self.tokenizer.batch_decode(comp_ids, skip_special_tokens=True)
        return comp_ids, comp_mask, texts

    # ---- one GRPO update over a single prompt ----
    def _train_one(self, question: str, gold: str | None):
        cfg = self.cfg
        prompt_ids, prompt_mask = _build_prompt_ids(self.tokenizer, question, cfg.system_prompt)
        prompt_len = prompt_ids.shape[1]

        comp_ids, comp_mask, texts = self._generate_group(prompt_ids, prompt_mask)
        G = comp_ids.shape[0]

        # rewards + group-relative advantage
        rewards = torch.tensor(
            [self.reward_fn(t, gold) for t in texts], dtype=torch.float32, device=self.device
        )
        mean_r = rewards.mean()
        adv = (rewards - mean_r) / (rewards.std(unbiased=False) + 1e-8)  # (G,)

        # assemble full sequences
        full_ids = torch.cat([prompt_ids.repeat(G, 1), comp_ids], dim=1)
        full_mask = torch.cat([prompt_mask.repeat(G, 1), comp_mask], dim=1)
        comp_full = torch.cat(
            [torch.zeros(G, prompt_len, dtype=torch.long, device=self.device), comp_mask], dim=1
        )

        pad = self.tokenizer.pad_token_id
        with torch.no_grad():
            old_tlp, comp_shift = _token_logprobs(self.model, full_ids, full_mask, comp_full, pad)
            ref_tlp, _ = _token_logprobs(self.ref_model, full_ids, full_mask, comp_full, pad)
        old_tlp = old_tlp.detach()
        ref_tlp = ref_tlp.detach()
        denom = comp_shift.sum().clamp(min=1.0)

        # inner optimisation loop (mu epochs re-using the same rollouts)
        metrics = {}
        for _ in range(cfg.mu):
            new_tlp, comp_shift2 = _token_logprobs(self.model, full_ids, full_mask, comp_full, pad)
            ratio = torch.exp(new_tlp - old_tlp)                       # (G, L-1)
            adv_b = adv.unsqueeze(1)                                   # (G, 1)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
            policy = -torch.min(surr1, surr2) * comp_shift2
            kl = (ref_tlp - new_tlp) * comp_shift2                    # per-token KL estimate
            loss = (policy + cfg.beta * kl).sum() / denom

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

        with torch.no_grad():
            new_r = torch.tensor([self.reward_fn(t, gold) for t in texts], device=self.device)
        metrics = {
            "loss": float(loss.detach().item()),
            "reward_mean": float(mean_r.item()),
            "reward_min": float(rewards.min().item()),
            "reward_max": float(rewards.max().item()),
            "adv_std": float(rewards.std(unbiased=False).item()),
        }
        return metrics, texts, new_r

    # ---- targeted SFT fallback (paper §3.2) ----
    def targeted_sft(self, examples: list[tuple[str, str]]) -> None:
        """A few supervised steps on curated (question, gold) pairs to unstick a stuck category."""
        cfg = self.cfg
        if not examples:
            return
        self.model.train()
        for q, gold in examples[: cfg.targeted_sft_steps]:
            ids, mask = _build_prompt_ids(self.tokenizer, q, cfg.system_prompt)
            tgt_text = f"<think>\n{gold}\n</think>\n\n{gold}"
            tgt_ids = self.tokenizer(tgt_text, return_tensors="pt").input_ids.to(self.device)
            full = torch.cat([ids, tgt_ids], dim=1)
            comp_mask = torch.cat(
                [torch.zeros(1, ids.shape[1], dtype=torch.long, device=self.device),
                 torch.ones(1, tgt_ids.shape[1], dtype=torch.long, device=self.device)],
                dim=1,
            )
            out = self.model(input_ids=full)
            logp = out.logits[:, :-1, :].log_softmax(-1)
            labels = full[:, 1:]
            tl = logp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            cm = comp_mask[:, 1:].to(tl.dtype)
            loss = -(tl * cm).sum() / cm.sum().clamp(min=1.0)
            self.model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
        logger.info("Targeted SFT applied on %d curated examples", min(len(examples), cfg.targeted_sft_steps))

    # ---- training loop ----
    def train(
        self,
        questions: list[str],
        golds: list[str | None],
        prompts_meta: list[dict] | None = None,
    ) -> None:
        cfg = self.cfg
        random.seed(cfg.seed)
        idx = list(range(len(questions)))

        total_steps = 0
        for epoch in range(cfg.epochs):
            random.shuffle(idx)
            for i in idx:
                self.model.train()
                metrics, texts, new_r = self._train_one(questions[i], golds[i])
                total_steps += 1

                # stall monitor -> targeted SFT (paper: switch when GRPO fails to converge)
                if metrics["reward_mean"] > self._best_reward + 1e-4:
                    self._best_reward = metrics["reward_mean"]
                    self._steps_since_improve = 0
                else:
                    self._steps_since_improve += 1
                if self._steps_since_improve >= cfg.stall_patience:
                    logger.warning(
                        "GRPO stalled %d steps (best=%.3f) -> running targeted SFT",
                        self._steps_since_improve, self._best_reward,
                    )
                    worst = sorted(range(len(texts)), key=lambda k: new_r[k].item())[: cfg.targeted_sft_steps]
                    curated = [(questions[i], golds[i] or extract_answer(texts[k])) for k in worst]
                    self.targeted_sft(curated)
                    self._steps_since_improve = 0

                if cfg.log_every and total_steps % cfg.log_every == 0:
                    logger.info(
                        "step %d | loss %.3f | reward %.3f (min %.2f max %.2f) | adv_std %.2f",
                        total_steps, metrics["loss"], metrics["reward_mean"],
                        metrics["reward_min"], metrics["reward_max"], metrics["adv_std"],
                    )


# ---------------------------------------------------------------------------
# Public API -- matches the existing stub signature
# ---------------------------------------------------------------------------
def train_stage2(
    hard_subset: str,
    output_dir: str,
    *,
    model_name: str = "Qwen/Qwen3.5-9B",
    stage1_adapter: str | None = None,
    cfg: GRPOConfig | None = None,
    max_samples: int | None = None,
    model_cfg: ModelConfig | None = None,
) -> None:
    """Run Stage-2 GRPO (+ targeted SFT fallback) on a *hard subset*.

    Args:
        hard_subset: JSONL of difficult examples, one per line, each with at least
            ``question`` (or ``query``) and ``answer`` (or ``gold``).  Optionally ``thinking``
            and a ``task`` tag (used only for logging).  These typically come from the
            attribution loop / Stage-1 error analysis (see ``attribution.py``).
        output_dir: directory for the saved Stage-2 LoRA adapter.
        model_name: base model id (default Qwen/Qwen3.5-9B). Overrides ``model_cfg``
            only when explicitly different from the default.
        stage1_adapter: optional path to a Stage-1 LoRA adapter to start from.
        cfg: GRPOConfig (group_size=8 by default).
        max_samples: cap on how many hard examples to use (prototype runs).
        model_cfg: shared ModelConfig (precision/dtype/quantisation). Honours the
            same ``precision`` switch as Stage 1 so both stages load the base model
            identically.
    """
    cfg = cfg or GRPOConfig()
    cfg.output_dir = output_dir  # type: ignore[attr-defined]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    _model_cfg = model_cfg or ModelConfig()
    if model_name != "Qwen/Qwen3.5-9B":
        _model_cfg.model_name_or_path = model_name

    tokenizer = load_tokenizer(_model_cfg.model_name_or_path)
    # current (trainable) policy
    model = load_model(_model_cfg.model_name_or_path, cfg=_model_cfg)
    model = apply_lora(model, cfg=_model_cfg)
    if stage1_adapter:
        model.load_adapter(stage1_adapter, adapter_name="stage1")
        logger.info("Loaded Stage-1 adapter from %s", stage1_adapter)
    # frozen reference policy (KL target) -- base model, no LoRA
    ref_model = load_model(_model_cfg.model_name_or_path, cfg=_model_cfg)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # load hard subset
    questions, golds = [], []
    with open(hard_subset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            q = ex.get("question") or ex.get("query")
            g = ex.get("answer") or ex.get("gold")
            if not q:
                continue
            questions.append(q)
            golds.append(g)
            if max_samples and len(questions) >= max_samples:
                break
    logger.info("Loaded %d hard examples from %s", len(questions), hard_subset)

    def reward_fn(text: str, gold: str | None) -> float:
        return composite_reward(text, gold, cfg.reward_weights)  # type: ignore[arg-type]

    trainer = GRPOTrainer(model, ref_model, tokenizer, cfg, reward_fn)
    trainer.train(questions, golds)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Stage-2 adapter saved -> %s", output_dir)


# ---------------------------------------------------------------------------
# CLI  (python -m grpo)
# Defaults come from config.yaml (this folder); CLI flags override the YAML values.
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    import argparse

    _DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "config.yaml")
    p = argparse.ArgumentParser(description="Stage-2 GRPO (+ targeted SFT) for Agentar-Fin-R1")
    p.add_argument("--config", default=_DEFAULT_CONFIG)
    p.add_argument("--hard-subset", default=None, help="JSONL of hard examples (question/answer)")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--stage1-adapter", default=None)
    p.add_argument("--group-size", type=int, default=None, help="G rollouts per prompt")
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

    # Build the shared ModelConfig — `precision` (fp16/bf16/int4) drives dtype + quantisation.
    model_cfg = ModelConfig(
        model_name_or_path=model.get("model_name", "Qwen/Qwen3.5-9B"),
        precision=model.get("precision", "fp16"),
    )

    train_stage2(
        hard_subset=hard_subset,
        output_dir=args.output_dir or run.get("output_dir", "checkpoints/stage2-grpo"),
        model_name=args.model_name or model.get("model_name", "Qwen/Qwen3.5-9B"),
        stage1_adapter=args.stage1_adapter or model.get("stage1_adapter"),
        cfg=cfg,
        max_samples=args.max_samples if args.max_samples is not None else data.get("max_samples"),
        model_cfg=model_cfg,
    )
