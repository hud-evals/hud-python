#!/usr/bin/env python3
"""Generic HuggingFace dataset evaluation runner with parallel execution support.

This script lets you evaluate any HUD-compatible Task dataset with either
Claude or OpenAI (Operator) agents. Supports both asyncio-based concurrency
(for small datasets) and process-based parallelization (for large datasets).

Prerequisites:
• `uv add hud-python`
• Set `HUD_API_KEY` in your env.
• Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your env.

Usage examples
──────────────
# Run a single OSWorld-Verified task with OpenAI Operator agent (default single-task mode)
python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent openai

# Same but with detailed agent step logs visible
python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent openai --verbose

# Enable debug-level logs for maximum visibility
python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent openai --very-verbose

# Evaluate the FULL SheetBench dataset with Claude (Single Process concurrency)
python examples/run_evaluation.py hud-evals/SheetBench-50 --full --agent claude --max-concurrent 25

# Run OSWorld-Verified dataset full with PARALLEL execution (300+ tasks)
python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent openai --full --parallel

# Parallel mode with manual configuration (8 workers, 10 concurrent per worker)
python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent openai --full --parallel --max-workers 8 --max-concurrent-per-worker 10

# Custom max steps per task (useful for complex tasks)
python examples/run_evaluation.py hud-evals/SheetBench-50 --full --max-steps 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Literal

import hud
from datasets import load_dataset
from hud.agents import ClaudeAgent, OperatorAgent
from hud.agents.misc.response_agent import ResponseAgent
from hud.clients import MCPClient
from hud.datasets import Task, run_dataset, run_dataset_parallel, run_dataset_parallel_manual

logger = logging.getLogger(__name__)

# Uncomment to enable logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s", datefmt="%H:%M:%S"
# )

# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------


def _build_agent(
    agent_type: Literal["claude", "openai"],
    *,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
) -> ClaudeAgent | OperatorAgent:
    """Create and return the requested agent type."""

    if agent_type == "openai":
        allowed_tools = allowed_tools or ["openai_computer"]

        return OperatorAgent(
            allowed_tools=allowed_tools,
            response_agent=ResponseAgent(),
        )

    # Fallback Claude agent (Anthropic)
    model = model or "claude-sonnet-4-20250514"
    allowed_tools = allowed_tools or ["anthropic_computer"]

    return ClaudeAgent(
        model=model,
        allowed_tools=allowed_tools,
        response_agent=ResponseAgent(),
    )


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------


async def run_single_task(
    dataset_name: str,
    *,
    agent_type: Literal["claude", "openai"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_steps: int = 10,
) -> None:
    """Load *one* task from *dataset_name* and execute it."""

    # Enable agent step logging for single task mode
    logging.getLogger("hud.agents").setLevel(logging.INFO)
    logging.getLogger("hud.agents.base").setLevel(logging.INFO)

    print("📊 Loading dataset…")
    dataset = load_dataset(dataset_name, split="train")

    # Get a simple task from dataset (Open last tab task)
    sample_task = dataset[1]  # type: ignore[index]
    task_prompt = sample_task.get("prompt", f"Task {sample_task.get('id', 0)}")  # type: ignore[attr-defined]

    with hud.trace(name=task_prompt):
        task = Task(**sample_task)  # type: ignore[arg-type]

        agent = _build_agent(
            agent_type,
            model=model,
            allowed_tools=allowed_tools,
        )
        print("Task prompt: ", task.prompt)
        result = await agent.run(task, max_steps=max_steps)
        print("✅ Reward:", result.reward)


# ---------------------------------------------------------------------------
# Full-dataset runner
# ---------------------------------------------------------------------------


async def run_full_dataset(
    dataset_name: str,
    *,
    agent_type: Literal["claude", "openai"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_concurrent: int = 30,
    max_steps: int = 10,
    parallel: bool = False,
    max_workers: int | None = None,
    max_concurrent_per_worker: int = 25,
) -> list[Any]:
    """Run evaluation across the entire dataset.

    Uses either asyncio-based run_dataset or process-based run_dataset_parallel
    depending on the parallel flag.
    """

    # Build agent class + config for run_dataset – we pass the *class* and a minimal
    # config dict, run_dataset will create a fresh agent per task.
    if agent_type == "openai":
        agent_class = OperatorAgent
        agent_config: dict[str, Any] = {
            "allowed_tools": allowed_tools or ["openai_computer"],
            "validate_api_key": False,
        }
    else:
        agent_class = ClaudeAgent
        agent_config = {
            "model": model or "claude-sonnet-4-20250514",
            "allowed_tools": allowed_tools or ["anthropic_computer"],
            "validate_api_key": False,
        }

    eval_name = f"Evaluation {dataset_name.split('/')[-1]}"

    if parallel:
        print(f"🚀 Running PARALLEL evaluation (workers: {max_workers or 'auto'})…")
        if max_workers is None:
            # Use auto-optimization (now the default run_dataset_parallel)
            return await run_dataset_parallel(
                name=eval_name,
                dataset=dataset_name,
                agent_class=agent_class,
                agent_config=agent_config,
                metadata={"dataset": dataset_name, "parallel": True},
                max_steps=max_steps,
                auto_respond=True,
            )
        else:
            # Use manual configuration
            return await run_dataset_parallel_manual(
                name=eval_name,
                dataset=dataset_name,
                agent_class=agent_class,
                agent_config=agent_config,
                max_workers=max_workers,
                max_concurrent_per_worker=max_concurrent_per_worker,
                metadata={"dataset": dataset_name, "parallel": True},
                max_steps=max_steps,
                auto_respond=True,
            )
    else:
        print(f"🚀 Running evaluation (max_concurrent: {max_concurrent})…")
        return await run_dataset(
            name=eval_name,
            dataset=dataset_name,
            agent_class=agent_class,
            agent_config=agent_config,
            max_concurrent=max_concurrent,
            metadata={"dataset": dataset_name},
            max_steps=max_steps,
            auto_respond=True,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # type: ignore[valid-type]
    parser = argparse.ArgumentParser(
        description="Evaluate HUD datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s hud-evals/SheetBench-50                    # Single task test
  %(prog)s hud-evals/SheetBench-50 --full             # Full dataset (<100 tasks)
  %(prog)s hud-evals/LargeDataset --full --parallel   # Large dataset (100+ tasks)
  %(prog)s hud-evals/SheetBench-50 --very-verbose     # Run with debug logs
        """,
    )

    parser.add_argument("dataset", help="HuggingFace dataset ID")
    parser.add_argument("--full", action="store_true", help="Run entire dataset")

    # Agent
    parser.add_argument("--agent", choices=["claude", "openai"], default="claude")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument(
        "--allowed-tools", dest="allowed_tools", help="Tool allowlist (comma-separated)"
    )

    # Concurrency
    parser.add_argument(
        "--max-concurrent",
        dest="max_concurrent",
        type=int,
        default=50,
        help="Max concurrent tasks (default: 50)",
    )

    # Task settings
    parser.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=10,
        help="Max steps per task (default: 10)",
    )

    # Parallel mode (100+ tasks)
    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel execution for large datasets"
    )
    parser.add_argument("--max-workers", dest="max_workers", type=int, help="Worker processes")
    parser.add_argument(
        "--max-concurrent-per-worker",
        dest="max_concurrent_per_worker",
        type=int,
        default=25,
        help="Max concurrent tasks per worker",
    )

    # Logging
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed agent step logs"
    )
    parser.add_argument(
        "--very-verbose",
        "-vv",
        action="store_true",
        help="Show debug-level logs for maximum visibility",
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.very_verbose:
        # Debug-level logs - maximum visibility
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s", datefmt="%H:%M:%S"
        )
        # Ensure HUD agent logs are at debug level
        logging.getLogger("hud.agents").setLevel(logging.DEBUG)
        logging.getLogger("hud.agents.base").setLevel(logging.DEBUG)
    elif args.verbose:
        # Detailed logs - show everything including agent steps
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s", datefmt="%H:%M:%S"
        )
        # Ensure HUD agent logs are visible
        logging.getLogger("hud.agents").setLevel(logging.INFO)
        logging.getLogger("hud.agents.base").setLevel(logging.INFO)

    allowed_tools = (
        [t.strip() for t in args.allowed_tools.split(",") if t.strip()]
        if args.allowed_tools
        else None
    )

    if args.full:
        import time

        start_time = time.time()

        results = await run_full_dataset(
            args.dataset,
            agent_type=args.agent,
            model=args.model,
            allowed_tools=allowed_tools,
            max_concurrent=args.max_concurrent,
            max_steps=args.max_steps,
            parallel=args.parallel,
            max_workers=args.max_workers,
            max_concurrent_per_worker=args.max_concurrent_per_worker,
        )

        elapsed = time.time() - start_time

        # Print statistics
        print("\n" + "=" * 50)
        print("📊 Evaluation Complete!")
        print("=" * 50)
        print(f"Total tasks: {len(results)}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Throughput: {len(results) / elapsed:.2f} tasks/second")

        if args.parallel:
            print(f"Execution mode: PARALLEL (workers: {args.max_workers or 'auto'})")
        else:
            print(f"Execution mode: ASYNCIO (max_concurrent: {args.max_concurrent})")

        # Count successes
        successful = sum(1 for r in results if getattr(r, "reward", 0) > 0)
        print(
            f"Successful tasks: {successful}/{len(results)} ({100 * successful / len(results):.1f}%)"
        )

    else:
        print(f"Execution mode: Single Task (max_steps: {args.max_steps})")
        await run_single_task(
            args.dataset,
            agent_type=args.agent,
            model=args.model,
            allowed_tools=allowed_tools,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    asyncio.run(main())
