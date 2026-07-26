"""Attribution loop: error -> (scene, task) -> data rollback / regeneration.

Writes attribution.json = [{label, pass@1, delta, eta, pi, allocated_samples}],
which the data loader reads to update sampling for the next round.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AttributionRecord:
    label: str
    pass_at_1: float
    delta: float
    eta: float
    pi: float
    allocated_samples: int


def write_attribution(records: list[AttributionRecord], out: str | Path) -> None:
    """Persist per-label attribution stats for the next training round."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.__dict__ for r in records]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def rollback_triggered(records: list[AttributionRecord], threshold: float = 0.0) -> bool:
    """True if any label's pass@1 dropped below its previous round (delta < threshold)."""
    return any(r.delta < threshold for r in records)
