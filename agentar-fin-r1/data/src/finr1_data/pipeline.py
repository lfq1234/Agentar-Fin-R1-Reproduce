"""Orchestrates the three-level data pipeline into golden training triplets.

Source -> Synthesis -> Verification -> (query, thinking, answer) golden data.
"""
from __future__ import annotations


def run(source_dir: str, out_dir: str) -> None:
    """Run the full data reproduction pipeline.

    Args:
        source_dir: directory of authoritative financial source documents.
        out_dir:    where golden triplets (Fin-R1-300K target) are written.

    TODO: wire the three stages defined in source/ synthesis/ verification/.
    """
    raise NotImplementedError("data pipeline not yet implemented")
