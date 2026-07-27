"""Source stage — Trusted Sources & Knowledge Engineering (paper §2.3.1).

Turns authoritative financial documents into a refined knowledge repository K of verified
knowledge units. Four sub-steps (paper, verbatim structure):

1. **Data Extraction**   — NER, dependency parsing, POS tagging → entities/relations/structures.
2. **Data Normalization**— unify heterogeneous formats, reconstruct structure.
3. **Data Detoxification**— remove non-compliant / contaminated / harmful content.
4. **Knowledge Refinement**— quality enhancement → high-fidelity refined knowledge units.

In this reproduction the *seed corpus* is the open ``antgroup/Agentar-DeepFinance-100K``
metadata + any local financial docs you drop in ``source_dir``. Steps 1–3 are implemented
with lightweight, dependency-free heuristics so the stage runs without spaCy/transformers;
swap in real NLP models by editing the clearly-marked TODOs.  Step 4 calls the LLM backend
to paraphrase/verify each unit (dry-run safe).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Iterable

from finr1_data.llm import LLMBackend, chat
from finr1_data.schema import KnowledgeUnit

logger = logging.getLogger(__name__)


# --- heuristics (replace with spaCy / transformers as needed) -----------------

_ENT_RE = re.compile(r"\b([A-Z][A-Za-z&]{2,}(?:\s[A-Z][A-Za-z&]{2,})*(?:\s(?:Inc|Corp|Ltd|Group|Bank|Co)\.?)?)\b")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*\s?%?")

# crude "toxic / non-compliant" keyword filter — placeholder for a real classifier
_TOXIC_TERMS = ("kill", "bomb", "insider trading tip", "guaranteed return", "100% profit")


def extract_entities(text: str) -> list[str]:
    """§2.3.1(1) NER stand-in: surface-form capitalization + number capture."""
    ents = list(dict.fromkeys(_ENT_RE.findall(text)))
    ents += [m.group() for m in _NUM_RE.finditer(text)]
    return ents


def normalize_record(raw: dict) -> dict:
    """§2.3.1(2) Normalization stand-in: trim, collapse whitespace, lowercase keys."""
    out = {}
    for k, v in raw.items():
        out[k.strip().lower()] = (v.strip() if isinstance(v, str) else v)
    return out


def is_detoxified(text: str) -> bool:
    """§2.3.1(3) Detoxification stand-in: keyword blocklist + empty check."""
    low = text.lower()
    if not text.strip():
        return False
    return not any(t in low for t in _TOXIC_TERMS)


def refine_unit(unit: KnowledgeUnit, backend: LLMBackend) -> KnowledgeUnit:
    """§2.3.1(4) Knowledge Refinement: LLM paraphrase/verify; fall back to copy."""
    sys = "You are a financial knowledge engineer. Rewrite the passage as one concise, "\
          "factual knowledge statement. Keep numbers exact. Reply with the statement only."
    try:
        improved = chat(sys, unit.text, backend).strip()
        if improved:
            unit.text = improved
    except Exception as exc:  # pragma: no cover - network guards
        logger.warning("refine_unit LLM call failed (%s); keeping original", exc)
    return unit


# --- seed loading ------------------------------------------------------------

def load_seed_records(source_dir: str) -> list[dict]:
    """Load raw records from *source_dir* (JSONL/JSON) — or empty if absent."""
    records: list[dict] = []
    if not source_dir or not os.path.isdir(source_dir):
        return records
    for name in sorted(os.listdir(source_dir)):
        path = os.path.join(source_dir, name)
        if name.endswith((".jsonl",)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif name.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.extend(data if isinstance(data, list) else [data])
    return records


def knowledge_engineering(
    records: Iterable[dict],
    backend: LLMBackend,
    *,
    lang: str = "en",
) -> list[KnowledgeUnit]:
    """Run the full §2.3.1 pipeline over *records* → refined knowledge repository K.

    Each record is normalized, detoxified, entities extracted, then refined via the LLM.
    Returns the list of :class:`KnowledgeUnit` ``k ∈ K``.
    """
    K: list[KnowledgeUnit] = []
    for i, rec in enumerate(records):
        text = rec.get("text") or rec.get("content") or rec.get("Solution") or ""
        if isinstance(text, list):
            text = " ".join(text)
        if not is_detoxified(text):
            continue
        norm = normalize_record(rec)
        entities = extract_entities(text)
        unit = KnowledgeUnit(
            id=f"k{i:08d}",
            text=text.strip(),
            scene=rec.get("scene") or norm.get("scene"),
            task=rec.get("task") or norm.get("task"),
            source_doc=rec.get("source") or rec.get("Content"),
            meta={"entities": entities, **{k: v for k, v in norm.items() if k not in ("text", "content")}},
        )
        unit = refine_unit(unit, backend)
        K.append(unit)
    logger.info("Knowledge engineering produced %d refined units (from %d records)", len(K), sum(1 for _ in records))
    return K


def save_knowledge_units(K: list[KnowledgeUnit], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u in K:
            f.write(json.dumps(u.__dict__, ensure_ascii=False) + "\n")
