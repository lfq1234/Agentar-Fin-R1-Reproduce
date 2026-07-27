"""Orchestrates the three-level data pipeline → golden (query, thinking, answer) triplets.

Flow (paper §2.3):

    Source  (knowledge_engineering)        → refined knowledge repository K  (§2.3.1)
      │
    Synthesis (union of)                                          (§2.3.2)
      ├─ Track I : task-oriented generation over K                (eq.2/3)
      ├─ Track II: self-evolution of instructions                 (eq.4/5)
      └─ MKE + CoT sampling + SCR  (DeepFinance-100K §3.3–3.5)    ← concrete seed generator
      │
    Verification (ensemble + rating + governance)                 (§2.3.3, eq.7–11)
      │
    D_final = { x ∈ D_synthesis | verify(x) ∧ clean(x) ∧ score(x) > τ }

The (query, thinking, answer) golden set is written to ``out_dir/golden.jsonl`` and can be
fed straight into the SFT stage via ``train_sft.py --extra-data <path>``.
"""
from __future__ import annotations

import json
import logging
import os

from finr1_data.llm import LLMBackend, get_backend
from finr1_data.schema import KnowledgeUnit, ReasoningTriplet
from finr1_data.source.knowledge_engineering import (
    knowledge_engineering,
    load_seed_records,
    save_knowledge_units,
)
from finr1_data.synthesis.task_oriented import track_i_task_oriented
from finr1_data.synthesis.self_evolution import track_ii_self_evolution
from finr1_data.synthesis.mke import (
    q2a_direct_curation,
    a2q_counterfactual,
    t2q_cot_mining,
    sample_cot,
    scr_rescue,
)
from finr1_data.verification.verify import verify_and_clean

logger = logging.getLogger(__name__)


# Tiny built-in seed used only when DeepFinance-100K cannot be loaded (e.g. dry-run with no
# `datasets` installed / offline). Clearly a placeholder — real runs pull the full 100K corpus.
_BUILTIN_SEED_QA: list[tuple[str, str]] = [
    ("What is the after-tax yield of a 5-year corporate bond priced at 98.5 with a 6% coupon "
     "and a 20% tax rate?", "The after-tax yield is approximately 4.87%."),
    ("A portfolio has two assets with weights 0.6 and 0.4 and returns 10% and 15%. What is the "
     "expected portfolio return?", "The expected portfolio return is 0.6*10% + 0.4*15% = 12%."),
    ("Explain the difference between a forward and a futures contract.",
     "Both are agreements to trade an asset at a future date; forwards are private and customizable "
     "(OTC), futures are exchange-traded and standardized."),
]


def _save_triplets(trips: list[ReasoningTriplet], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in trips:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def _extract_seed(rows: list[dict], max_seed: int | None) -> list[tuple[str, str]]:
    """Pull (question, answer) pairs out of raw dataset rows."""
    out: list[tuple[str, str]] = []
    for i, ex in enumerate(rows):
        if max_seed and i >= max_seed:
            break
        q = ex.get("Question") or ex.get("question") or ex.get("query")
        a = ex.get("Answer") or ex.get("answer") or ""
        sol = ex.get("Solution") or ex.get("solution") or ""
        if not a and sol:
            a = sol.split("Answer:")[-1].strip() if "Answer:" in sol else sol
        if q and a:
            out.append((str(q), str(a)))
    return out


def _load_deepfinance_seed_qa(max_seed: int | None = None,
                               local_path: str | None = None,
                               allow_builtin: bool = False) -> list[tuple[str, str]]:
    """Load (question, answer) seed pairs — the **seed corpus** for synthesis.

    Per paper §4.2 DeepFinance-100K is part of the training data and, per the dataset paper
    §3.2, it is exactly the seed corpus that MKE/CoT/SCR operate on.  We load it from a local
    copy first (``local_path``), then fall back to the HuggingFace hub.  Both paths degrade
    gracefully so the pipeline still runs on ``source_dir`` records alone.

    Requires ``pip install datasets`` for either path; if absent we return [] and log a hint
    (or a tiny built-in seed in dry-run so the pipeline stays runnable).
    """
    rows: list[dict] = []

    # 1) local copy (parquet / json / jsonl) — preferred, offline-friendly
    if local_path:
        try:
            from datasets import load_dataset as _ld
            if str(local_path).endswith((".json", ".jsonl")):
                rows = list(_ld("json", data_files=local_path, split="train"))
            else:
                rows = list(_ld("parquet", data_files=local_path, split="train"))
            logger.info("Loaded %d seed QA from local %s", len(rows), local_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local seed load failed (%s)", exc)

    # 2) HuggingFace hub
    if not rows:
        try:
            from datasets import load_dataset as _ld
        except ImportError:
            if allow_builtin:
                logger.warning("`datasets` not installed — using built-in dry-run seed "
                               "(%d pairs). Install `datasets` for the full 100K corpus.", len(_BUILTIN_SEED_QA))
                return _BUILTIN_SEED_QA[:max_seed] if max_seed else _BUILTIN_SEED_QA
            logger.warning("`datasets` not installed — skipping DeepFinance-100K seed "
                           "(pip install datasets). Pipeline runs on local source_dir only.")
            return []
        try:
            rows = list(_ld("antgroup/Agentar-DeepFinance-100K", split="train"))
            logger.info("Loaded %d DeepFinance-100K seed QA from hub", len(rows))
        except Exception as exc:  # noqa: BLE001
            if allow_builtin:
                logger.warning("Could not load DeepFinance-100K (%s) — using built-in dry-run seed.", exc)
                return _BUILTIN_SEED_QA[:max_seed] if max_seed else _BUILTIN_SEED_QA
            logger.warning("Could not load DeepFinance-100K (%s) — using local seed only", exc)
            return []

    return _extract_seed(rows, max_seed)


def run(
    source_dir: str,
    out_dir: str,
    *,
    backend_mode: str = "dry-run",
    backend_kwargs: dict | None = None,
    max_seed: int | None = None,
    tau: float = 0.5,
    benchmark_queries: list[str] | None = None,
) -> str:
    """Run the full data reproduction pipeline.

    Args:
        source_dir:       dir of authoritative financial source docs (JSON/JSONL). May be empty;
                          DeepFinance-100K is used as the concrete seed corpus regardless.
        out_dir:          output dir; writes ``knowledge.jsonl`` + ``golden.jsonl``.
        backend_mode:     "dry-run" (default, no API) | "openai" | "hf".
        backend_kwargs:   extra kwargs for the backend (e.g. model/base_url).
        max_seed:         cap on seed records (prototype subsample).
        tau:              eq.11 quality threshold.
        benchmark_queries: eval queries to decontaminate against.

    Returns:
        Path to the golden triplets JSONL.
    """
    backend = get_backend(backend_mode, **(backend_kwargs or {}))
    backends = [backend]  # ensemble of size ≥1 (grow with more backends for real consistency)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Stage 1: Source (§2.3.1) ----------
    logger.info("[1/3] Source: knowledge engineering")
    seed = load_seed_records(source_dir)
    if max_seed and len(seed) > max_seed:
        seed = seed[:max_seed]
    K = knowledge_engineering(seed, backend)
    save_knowledge_units(K, os.path.join(out_dir, "knowledge.jsonl"))
    logger.info("  → %d knowledge units", len(K))

    # ---------- Stage 2: Synthesis (§2.3.2 + DeepFinance-100K §3) ----------
    logger.info("[2/3] Synthesis: Track I + Track II + MKE/CoT/SCR")
    D: list[ReasoningTriplet] = []

    # Track I — task-oriented over K
    if K:
        D += track_i_task_oriented(K, backend, n_per_unit=1)

    # DeepFinance-100K as the concrete seed corpus → MKE + CoT + SCR
    seed_qa = _load_deepfinance_seed_qa(
        max_seed=max_seed,
        local_path=os.environ.get("DEEPFINANCE_LOCAL"),
        allow_builtin=(backend_mode == "dry-run"),
    )
    if seed_qa:
        qa_pairs = [(q, a) for q, a in seed_qa]
        D += q2a_direct_curation([{"question": q, "answer": a} for q, a in qa_pairs])
        D += a2q_counterfactual(qa_pairs, backend)
        D += sample_cot(qa_pairs, backend, n_samples=2)
        D += scr_rescue(qa_pairs, backend)            # rescue hard QA that failed sampling

    # Track II — self-evolution from a seed query set
    seed_queries = [q for q, _ in seed_qa[:50]] or [t.query for t in D[:50]]
    if seed_queries:
        D += track_ii_self_evolution(seed_queries, backend, max_iter=1)

    logger.info("  → %d raw synthesized triplets", len(D))

    # ---------- Stage 3: Verification & Governance (§2.3.3) ----------
    logger.info("[3/3] Verification: ensemble + rating + governance")
    final = verify_and_clean(D, backends, tau=tau, benchmark_queries=benchmark_queries or [])

    golden_path = os.path.join(out_dir, "golden.jsonl")
    _save_triplets(final, golden_path)
    logger.info("Pipeline done → %d golden triplets at %s", len(final), golden_path)
    return golden_path


def main() -> None:
    """CLI entry: ``python -m finr1_data.pipeline``."""
    import argparse

    p = argparse.ArgumentParser(description="Agentar-Fin-R1 data pipeline (§2.3)")
    p.add_argument("--source-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/golden")
    p.add_argument("--backend", default="dry-run", choices=["dry-run", "openai", "hf"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--backend-kwargs", default="{}",
                   help='JSON dict, e.g. \'{"base_url": "http://localhost:8000/v1"}\'')
    p.add_argument("--max-seed", type=int, default=None)
    p.add_argument("--tau", type=float, default=0.5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import json as _json
    backend_kwargs = _json.loads(args.backend_kwargs)
    if args.model:
        backend_kwargs.setdefault("model", args.model)
    golden = run(source_dir=args.source_dir, out_dir=args.out_dir,
                 backend_mode=args.backend, backend_kwargs=backend_kwargs,
                 max_seed=args.max_seed, tau=args.tau)
    print(f"\nGolden triplets → {golden}")


if __name__ == "__main__":
    main()
