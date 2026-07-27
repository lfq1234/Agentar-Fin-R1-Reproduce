"""CLI: run the data reproduction pipeline.

Examples
--------
    # dry-run (no API key) — validates wiring/schema on a tiny seed
    python -m finr1_data.pipeline --out-dir data/golden --max-seed 20

    # real generation via an OpenAI-compatible server
    python -m finr1_data.pipeline --backend openai --model Qwen/Qwen3-8B \
        --backend-kwargs '{"base_url": "http://localhost:8000/v1"}' --max-seed 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Agentar-Fin-R1 data pipeline (§2.3)")
    p.add_argument("--source-dir", default="data/raw",
                   help="dir of authoritative financial source docs (JSON/JSONL)")
    p.add_argument("--out-dir", default="data/golden")
    p.add_argument("--backend", default="dry-run", choices=["dry-run", "openai", "hf"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--backend-kwargs", default="{}",
                   help='JSON dict, e.g. \'{"base_url": "http://localhost:8000/v1"}\'')
    p.add_argument("--max-seed", type=int, default=None, help="cap seed records for prototype")
    p.add_argument("--tau", type=float, default=0.5, help="eq.11 quality threshold")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from finr1_data.pipeline import run

    backend_kwargs = json.loads(args.backend_kwargs)
    if args.model:
        backend_kwargs.setdefault("model", args.model)

    golden = run(
        source_dir=args.source_dir,
        out_dir=args.out_dir,
        backend_mode=args.backend,
        backend_kwargs=backend_kwargs,
        max_seed=args.max_seed,
        tau=args.tau,
    )
    print(f"\nGolden triplets → {golden}")


if __name__ == "__main__":
    main()
