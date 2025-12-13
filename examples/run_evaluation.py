#!/usr/bin/env python3
"""Example: Running evaluations programmatically with run_dataset.

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


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on a HUD dataset")
    parser.add_argument("dataset", help="Dataset source (e.g., hud-evals/SheetBench-50)")
    parser.add_argument("--agent", choices=["claude", "operator"], default="claude")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--max-concurrent", type=int, default=30, help="Max concurrent tasks")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per task")
    parser.add_argument("--group-size", type=int, default=1, help="Runs per task (for variance)")
    parser.add_argument("--task-ids", nargs="*", help="Specific task indices to run (optional)")
    args = parser.parse_args()

    # Import here to avoid import errors if agents not installed
    from hud.datasets import load_dataset, run_dataset, display_results

    # Load dataset as Task objects
    print(f"Loading {args.dataset}...")
    tasks = load_dataset(args.dataset)

    # Filter by index if specified
    if args.task_ids:
        indices = [int(tid) for tid in args.task_ids]
        tasks = [tasks[i] for i in indices if i < len(tasks)]
        print(f"Filtered to {len(tasks)} tasks at indices: {args.task_ids}")

    # Create agent instance based on type
    if args.agent == "operator":
        from hud.agents import OperatorAgent

        agent = OperatorAgent.create(
            checkpoint_name=args.model or "computer-use-preview",
        )
    else:
        from hud.agents import ClaudeAgent

        agent = ClaudeAgent.create(
            checkpoint_name=args.model or "claude-sonnet-4-5",
        )

    # Run evaluation
    print(f"Running {len(tasks)} tasks with {args.agent} agent...")
    results = await run_dataset(
        tasks=tasks,
        agent=agent,
        max_steps=args.max_steps,
        max_concurrent=args.max_concurrent,
        group_size=args.group_size,
    )

    # Display results
    print(f"\n{'=' * 50}")
    print(f"Completed {len(results)} tasks")
    for i, ctx in enumerate(results):
        reward = ctx.reward if hasattr(ctx, "reward") else "N/A"
        print(f"  Task {i}: reward={reward}")


if __name__ == "__main__":
    asyncio.run(main())
