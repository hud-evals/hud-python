#!/usr/bin/env python3
"""Example: Running evaluations programmatically with run_tasks.

For CLI usage, prefer `hud eval` which handles config files, interactive
agent selection, and more. This example shows the programmatic API.

Usage:
    python examples/run_evaluation.py hud-evals/SheetBench-50
    python examples/run_evaluation.py hud-evals/SheetBench-50 --agent claude --max-concurrent 50
    python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent operator
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, cast

from datasets import load_dataset

from hud.agents import ClaudeAgent, OperatorAgent
from hud.datasets import run_tasks, display_results
from hud.types import Task


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on a HUD dataset")
    parser.add_argument("dataset", help="HuggingFace dataset ID (e.g., hud-evals/SheetBench-50)")
    parser.add_argument("--agent", choices=["claude", "operator"], default="claude")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Max concurrent tasks")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per task")
    parser.add_argument("--group-size", type=int, default=1, help="Runs per task (for variance)")
    parser.add_argument("--task-ids", nargs="*", help="Specific task IDs to run (optional)")
    args = parser.parse_args()

    # Load dataset and convert to Task objects
    print(f"Loading {args.dataset}...")
    raw_dataset = load_dataset(args.dataset, split="train")
    tasks = [Task(**cast("dict[str, Any]", row)) for row in raw_dataset]

    # Filter by task IDs if specified
    if args.task_ids:
        tasks = [t for t in tasks if t.id in args.task_ids]
        print(f"Filtered to {len(tasks)} tasks: {args.task_ids}")

    # Select agent class and config
    if args.agent == "operator":
        agent_class = OperatorAgent
        agent_config = {"model": args.model or "computer-use-preview", "validate_api_key": False}
    else:
        agent_class = ClaudeAgent
        agent_config = {"model": args.model or "claude-sonnet-4-5", "validate_api_key": False}

    # Run evaluation
    results = await run_tasks(
        tasks=tasks,
        agent_class=agent_class,
        agent_config=agent_config,
        name=f"Eval: {args.dataset.split('/')[-1]}",
        max_concurrent=args.max_concurrent,
        max_steps=args.max_steps,
        group_size=args.group_size,
        auto_respond=True,
    )

    # Display results (works for both single and grouped runs)
    display_results(results)


if __name__ == "__main__":
    asyncio.run(main())
