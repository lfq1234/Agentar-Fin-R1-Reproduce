"""Agent runtime: plans and executes financial tasks.

Mirrors the Agent capabilities evaluated by Finova:
intent detection, slot filling, tool planning, expression generation.

Currently a stub that echoes the dispatched (scene, task) context. Wire it to
`app.services.inference` (the reproduced model) and a tool registry to make it real.
"""
from __future__ import annotations

from typing import Any


def run_agent(
    message: str,
    scene: str | None = None,
    task: str | None = None,
    structured: bool = False,
) -> Any:
    # TODO: call services.inference.generate(...) with the reproduced model,
    # then route through the tool registry (financial data / calculation / compliance).
    if structured:
        return {
            "intent": None,
            "slots": {},
            "tool_plan": [],
            "expression": f"[stub] {message}",
        }
    return f"[stub] dispatched(scene={scene}, task={task}): {message}"
