"""CLI entry point: ``python -m finr1_training.scripts.train_sft``

Loads config from ``configs/sft_stage1.yaml`` (or CLI overrides) and launches
Stage 1 SFT training (paper §3.2).  The financial CoT corpus is DeepFinance-100K
(paper §4.2 training data); the (query,thinking,answer) ternary synthesis lives in
the data pipeline (``agentar-fin-r1/data``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package is importable when running as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 SFT — Agentar-Fin-R1 reproduction (paper §3.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default=str(_PROJECT_ROOT / "configs" / "sft_stage1.yaml"),
        help="Path to YAML config file",
    )
    # Common overrides
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--financial-data", type=str, default=None,
                        help="Financial CoT corpus (DeepFinance-100K by default)")
    parser.add_argument("--extra-data", type=str, default=None,
                        help="JSONL from data pipeline (ternary-group synthesis) to merge")
    parser.add_argument("--general-data", type=str, default=None)
    parser.add_argument("--max-financial", type=int, default=None)
    parser.add_argument("--max-general", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--weighting", choices=["complexity", "heuristic", "passk"], default=None)
    parser.add_argument("--no-thinking", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    d = cfg.get("data", {})
    dw = cfg.get("difficulty_weighting", {})
    t = cfg.get("training", {})

    # Import here so --help works without torch installed
    from finr1_training.stage1_sft import train_stage1

    sft_overrides: dict = {}
    sft_overrides["num_train_epochs"] = args.epochs or t.get("num_train_epochs", 3)
    sft_overrides["per_device_train_batch_size"] = args.batch_size or t.get("per_device_train_batch_size", 4)
    sft_overrides["learning_rate"] = args.lr or t.get("learning_rate", 2e-4)
    sft_overrides["max_seq_length"] = args.seq_length or d.get("max_seq_length", 4096)

    ckpt = train_stage1(
        output_dir=args.output_dir or cfg.get("output_dir", "./checkpoints/stage1"),
        financial_data=args.financial_data or d.get("financial_data", "antgroup/Agentar-DeepFinance-100K"),
        extra_data_path=args.extra_data or d.get("extra_data"),
        general_data=args.general_data or d.get("general_data"),
        max_financial=args.max_financial or d.get("max_financial"),
        max_general=args.max_general or d.get("max_general"),
        include_thinking=not args.no_thinking if args.no_thinking else d.get("include_thinking", True),
        max_seq_length=sft_overrides["max_seq_length"],
        weighting_method=args.weighting or dw.get("method", "complexity"),
        sft_args_override=sft_overrides,
    )
    print(f"\n{'='*60}")
    print(f"Stage 1 SFT complete!")
    print(f"Checkpoint: {ckpt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
