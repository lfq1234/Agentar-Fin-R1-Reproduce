"""Label System for Agentar-Fin-R1 reproduction.

Two-dimensional, non-orthogonal labels: (Scene, Task).
Scene : Banking, Securities, Insurance, Trust, MutualFunds
Task  : NER, IntentClassification, SlotFilling, EntityDisambiguation, ConsultationQA

非正交稀疏：并非所有 Task 适用于所有 Scene，真实还原金融任务分布。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SCENES: Final[list[str]] = [
    "Banking",
    "Securities",
    "Insurance",
    "Trust",
    "MutualFunds",
]

TASKS: Final[list[str]] = [
    "NER",
    "IntentClassification",
    "SlotFilling",
    "EntityDisambiguation",
    "ConsultationQA",
]

# 非正交：每个 Scene 适用的 Task 子集。
APPLICABLE: Final[dict[str, list[str]]] = {
    "Banking": ["NER", "IntentClassification", "SlotFilling", "ConsultationQA"],
    "Securities": [
        "NER",
        "IntentClassification",
        "SlotFilling",
        "EntityDisambiguation",
        "ConsultationQA",
    ],
    "Insurance": ["NER", "IntentClassification", "SlotFilling", "ConsultationQA"],
    "Trust": ["NER", "EntityDisambiguation", "ConsultationQA"],
    "MutualFunds": ["NER", "IntentClassification", "SlotFilling", "ConsultationQA"],
}


@dataclass(frozen=True)
class Label:
    scene: str
    task: str

    def __post_init__(self) -> None:
        if self.scene not in SCENES:
            raise ValueError(f"unknown scene: {self.scene}")
        if self.task not in TASKS:
            raise ValueError(f"unknown task: {self.task}")
        if self.task not in APPLICABLE.get(self.scene, []):
            raise ValueError(
                f"task {self.task!r} not applicable to scene {self.scene!r}"
            )


def all_labels() -> list[Label]:
    """Enumerate every valid (scene, task) label."""
    return [Label(s, t) for s in SCENES for t in APPLICABLE.get(s, [])]
