"""finr1_data: data reproduction pipeline for Agentar-Fin-R1.

Three-level governance (paper §2.3): Source -> Synthesis -> Verification,
producing (query, thinking, answer) golden triplets.

Quick start (dry-run, no API key)::

    python -m finr1_data.pipeline --out-dir data/golden --max-seed 20
"""
from __future__ import annotations

from .schema import KnowledgeUnit, ReasoningTriplet
from .pipeline import run

__all__ = ["run", "KnowledgeUnit", "ReasoningTriplet"]
