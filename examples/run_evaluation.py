#!/usr/bin/env python3
"""Generic HuggingFace dataset evaluation runner using asyncio-based concurrency.

This script lets you evaluate any HUD-compatible Task dataset with either
Claude or OpenAI (Operator) agents using efficient asyncio-based concurrency.

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

# Evaluate the FULL SheetBench dataset with Claude
python examples/run_evaluation.py hud-evals/SheetBench-50 --full --agent claude --max-concurrent 50

# Run OSWorld-Verified dataset with higher concurrency for speed
python examples/run_evaluation.py hud-evals/OSWorld-Verified-Gold --agent openai --full --max-concurrent 100

# Limit concurrency to prevent rate limits
python examples/run_evaluation.py hud-evals/SheetBench-50 --agent openai --full --max-concurrent 20

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
from hud.datasets import Task, run_dataset
from hud.types import AgentType

logger = logging.getLogger(__name__)

# Uncomment to enable logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s", datefmt="%H:%M:%S"
)

# ---------------------------------------------------------------------------
# Agent factory helpers
# ---------------------------------------------------------------------------


def _build_agent(
    agent_type: Literal[AgentType.CLAUDE, AgentType.OPERATOR],
    *,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
) -> ClaudeAgent | OperatorAgent:
    """Create and return the requested agent type."""
    if agent_type == AgentType.OPERATOR:
        return OperatorAgent(allowed_tools=allowed_tools, validate_api_key=False)

    model = model or "claude-sonnet-4-5-20250929"
    return ClaudeAgent(model=model, allowed_tools=allowed_tools, validate_api_key=False)


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------


async def run_single_task(
    dataset_name: str,
    *,
    agent_type: Literal[AgentType.CLAUDE, AgentType.OPERATOR] = AgentType.CLAUDE,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_steps: int = 10,
) -> None:
    """Load one task from dataset and execute it."""
    logging.getLogger("hud.agents").setLevel(logging.INFO)
    logging.getLogger("hud.agents.base").setLevel(logging.INFO)

    print("📊 Loading dataset…")
    dataset = load_dataset(dataset_name, split="train")

    sample_task = dataset[1]  # type: ignore[index]
    task_prompt = sample_task.get("prompt", f"Task {sample_task.get('id', 0)}")  # type: ignore[attr-defined]

    async with hud.async_trace(name=task_prompt):
        task = Task(**sample_task)  # type: ignore[arg-type]
        agent = _build_agent(agent_type, model=model, allowed_tools=allowed_tools)
        print("Task prompt: ", task.prompt)
        result = await agent.run(task, max_steps=max_steps)
        print("✅ Reward:", result.reward)


# ---------------------------------------------------------------------------
# Full-dataset runner
# ---------------------------------------------------------------------------


async def run_full_dataset(
    dataset_name: str,
    *,
    agent_type: Literal[AgentType.CLAUDE, AgentType.OPERATOR] = AgentType.CLAUDE,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_concurrent: int = 50,
    max_steps: int = 10,
) -> list[Any]:
    """Run evaluation across entire dataset with asyncio concurrency."""
    if agent_type == AgentType.OPERATOR:
        agent_class = OperatorAgent
        agent_config = {"validate_api_key": False}
        if model:
            agent_config["model"] = model
        if allowed_tools:
            # Only pass allowed tools if they are provided, otherwise all tools are enabled
            agent_config["allowed_tools"] = allowed_tools
    else:
        agent_class = ClaudeAgent
        agent_config = {
            "model": model or "claude-sonnet-4-5-20250929",
            "validate_api_key": False,
        }
        if allowed_tools:
            # Only pass allowed tools if they are provided, otherwise all tools are enabled
            agent_config["allowed_tools"] = allowed_tools

    eval_name = f"Evaluation {dataset_name.split('/')[-1]}"

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
    parser.add_argument("--agent", choices=["claude", "operator"], default="claude")
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
        help="Max concurrent tasks (1-200 recommended, default: 50)",
    )

    # Task settings
    parser.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=10,
        help="Max steps per task (default: 10)",
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
        )

        elapsed = time.time() - start_time

        print("\n" + "=" * 50)
        print("📊 Evaluation Complete!")
        print("=" * 50)
        print(f"Total tasks: {len(results)}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        print(f"Throughput: {len(results) / elapsed:.2f} tasks/second")
        print(f"Execution mode: ASYNCIO (max_concurrent: {args.max_concurrent})")

        successful = sum(1 for r in results if getattr(r, "reward", 0) > 0.7)
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
