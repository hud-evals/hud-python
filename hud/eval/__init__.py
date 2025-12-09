"""HUD Eval - Evaluation context and management.

This module provides:
- EvalContext: Environment with evaluation tracking (trace_id, reward, etc.)
- EvalMixin: Adds env.eval() method to Environment
- eval(): Standalone context manager for task-based evaluation

Usage:
    # Method on existing environment
    async with env.eval("task_name") as env:
        await env.call_tool("navigate", url="...")
        env.reward = 0.9

    # Standalone with task slugs
    async with hud.eval("my-org/task:1") as env:
        await agent.run(env)

    # Blank eval for manual reward
    async with hud.eval() as env:
        env.reward = compute_reward()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# EvalMixin is safe to import (uses lazy imports internally)
from hud.eval.mixin import EvalMixin

# run_eval is safe to import (uses lazy imports internally)
from hud.eval.manager import run_eval

if TYPE_CHECKING:
    from hud.eval.context import EvalContext

__all__ = [
    "EvalContext",
    "EvalMixin",
    "run_eval",
]


def __getattr__(name: str) -> object:
    """Lazy import EvalContext to avoid circular imports."""
    if name == "EvalContext":
        from hud.eval.context import EvalContext
        return EvalContext
    raise AttributeError(f"module 'hud.eval' has no attribute {name!r}")
