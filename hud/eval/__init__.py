"""HUD Eval - Evaluation context and management.

This module provides:
- Task: A runnable evaluation unit (from env())
- EvalContext: Environment with evaluation tracking (trace_id, reward, etc.)
- eval(): Standalone context manager for task-based evaluation

Usage:
    # Using env() to create Task
    env = Environment("my-env").connect_hub("browser")

    async with env() as ctx:
        await ctx.call_tool("navigate", url="...")

    async with env("checkout", user_id="alice") as ctx:
        await ctx.submit("answer")

    # Orchestrated with Task objects
    tasks = [env("checkout", user_id="alice"), env("checkout", user_id="bob")]
    async with hud.eval(tasks, variants={"model": ["gpt-4o"]}, group=4) as ctx:
        await ctx._run(agent)

    # Blank eval for manual reward
    async with hud.eval() as ctx:
        ctx.reward = compute_reward()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# run_eval is safe to import (uses lazy imports internally). HTTP
# auto-instrumentation is applied lazily via hud._runtime.activate_runtime(),
# not on import.
from hud.eval.manager import run_eval

if TYPE_CHECKING:
    from hud.eval.context import EvalContext
    from hud.eval.task import Task

__all__ = [
    "EvalContext",
    "Task",
    "run_eval",
]


def __getattr__(name: str) -> object:
    """Lazily import EvalContext / Task.

    Keeping ``Task`` lazy avoids eagerly importing ``hud.eval.task`` during
    ``hud.eval`` package import, which would otherwise re-enter the
    ``hud.types`` <-> ``hud.eval.task`` cycle before ``hud.types`` finishes
    initializing.
    """
    if name == "EvalContext":
        from hud.eval.context import EvalContext

        return EvalContext
    if name == "Task":
        from hud.eval.task import Task

        return Task
    raise AttributeError(f"module 'hud.eval' has no attribute {name!r}")
