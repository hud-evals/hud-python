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
        await agent.run(ctx.prompt)

    # Standalone with task slugs
    async with hud.eval("my-org/task:1") as ctx:
        await agent.run(ctx)

    # Orchestrated with Task objects
    tasks = [env("checkout", user_id="alice"), env("checkout", user_id="bob")]
    async with hud.eval(tasks, variants={"model": ["gpt-4o"]}, group=4) as ctx:
        await agent.run(ctx.prompt)

    # Blank eval for manual reward
    async with hud.eval() as ctx:
        ctx.reward = compute_reward()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Auto-instrument httpx on import
import hud.eval.instrument  # noqa: F401

# run_eval is safe to import (uses lazy imports internally)
from hud.eval.manager import run_eval

# Task is safe to import
from hud.eval.task import Task

# Utils for v4 format handling
from hud.eval.utils import build_env_from_v4, is_v4_format, validate_v4_task

if TYPE_CHECKING:
    from hud.eval.context import EvalContext

__all__ = [
    "EvalContext",
    "Task",
    "build_env_from_v4",
    "is_v4_format",
    "run_eval",
    "validate_v4_task",
]


def __getattr__(name: str) -> object:
    """Lazy import EvalContext to avoid circular imports."""
    if name == "EvalContext":
        from hud.eval.context import EvalContext

        return EvalContext
    raise AttributeError(f"module 'hud.eval' has no attribute {name!r}")
