"""HUD evaluation command for running tasks and datasets."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal

import typer

import hud
from hud.utils.hud_console import HUDConsole

logger = logging.getLogger(__name__)
hud_console = HUDConsole()


def build_agent(
    agent_type: Literal["claude", "openai"],
    *,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    verbose: bool = False,
) -> Any:
    """Create and return the requested agent type."""

    # Import agents lazily to avoid dependency issues
    if agent_type == "openai":
        try:
            from hud.agents import OperatorAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        if allowed_tools:
            return OperatorAgent(
                allowed_tools=allowed_tools,
                verbose=verbose,
            )
        else:
            return OperatorAgent(verbose=verbose)

    # Fallback Claude agent (Anthropic)
    try:
        from hud.agents import ClaudeAgent
    except ImportError as e:
        hud_console.error(
            "Claude agent dependencies are not installed. "
            "Please install with: pip install 'hud-python[agent]'"
        )
        raise typer.Exit(1) from e

    model = model or "claude-sonnet-4-20250514"

    if allowed_tools:
        return ClaudeAgent(
            model=model,
            allowed_tools=allowed_tools,
            verbose=verbose,
        )
    else:
        return ClaudeAgent(
            model=model,
            verbose=verbose,
        )


async def run_single_task(
    source: str,
    *,
    agent_type: Literal["claude", "openai"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_steps: int = 10,
    verbose: bool = False,
) -> None:
    """Load one task and execute it, or detect if JSON contains a list and run as dataset."""

    # Import Task and run_dataset lazily
    try:
        from hud.datasets import Task, run_dataset
    except ImportError as e:
        hud_console.error(
            "Dataset dependencies are not installed. "
            "Please install with: pip install 'hud-python\u27e6agent\u27e7'"
        )
        raise typer.Exit(1) from e

    # Check if it's a JSON file
    path = Path(source)
    if path.exists() and path.suffix == ".json":
        hud_console.info("📊 Loading task file…")
        with open(path) as f:  # noqa: ASYNC230
            json_data = json.load(f)

        # Check if JSON contains multiple tasks (list with more than 1 task)
        if isinstance(json_data, list) and len(json_data) > 1:
            hud_console.info(f"Found {len(json_data)} tasks in JSON file, running as dataset…")

            # Build agent class and config for run_dataset
            if agent_type == "openai":
                try:
                    from hud.agents import OperatorAgent

                    agent_class = OperatorAgent
                except ImportError as e:
                    hud_console.error(
                        "OpenAI agent dependencies are not installed. "
                        "Please install with: pip install 'hud-python\u27e6agent\u27e7'"
                    )
                    raise typer.Exit(1) from e

                agent_config: dict[str, Any] = {"verbose": verbose}
                if allowed_tools:
                    agent_config["allowed_tools"] = allowed_tools

            else:
                try:
                    from hud.agents import ClaudeAgent

                    agent_class = ClaudeAgent
                except ImportError as e:
                    hud_console.error(
                        "Claude agent dependencies are not installed. "
                        "Please install with: pip install 'hud-python[agent]'"
                    )
                    raise typer.Exit(1) from e

                agent_config = {
                    "model": model or "claude-sonnet-4-20250514",
                    "verbose": verbose,
                }
                if allowed_tools:
                    agent_config["allowed_tools"] = allowed_tools

            # Run as dataset with single-task concurrency to maintain debug behavior
            results = await run_dataset(
                name=f"JSON Dataset: {path.name}",
                dataset=json_data,  # Pass the list directly
                agent_class=agent_class,
                agent_config=agent_config,
                max_concurrent=1,  # Run sequentially for debug mode
                metadata={"source": str(path)},
                max_steps=max_steps,
            )

            # Display summary
            successful = sum(1 for r in results if getattr(r, "reward", 0) > 0)
            hud_console.success(f"Completed {len(results)} tasks: {successful} successful")
            return

        # Single task JSON (either direct object or list with 1 task)
        if isinstance(json_data, list) and len(json_data) == 1:
            hud_console.info("Found 1 task in JSON file, running as single task…")
            task = Task(**json_data[0])
        elif isinstance(json_data, dict):
            task = Task(**json_data)
        else:
            hud_console.error("JSON file must contain a list of tasks when using --full flag")
            raise typer.Exit(1)
    else:
        # Load from HuggingFace dataset
        hud_console.info(f"📊 Loading dataset from HuggingFace: {source}…")
        try:
            from datasets import load_dataset
        except ImportError as e:
            hud_console.error(
                "Datasets library is not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        dataset = load_dataset(source, split="train")

        # Get first task from dataset
        sample_task = dataset[0]  # type: ignore[index]
        task = Task(**sample_task)  # type: ignore[arg-type]

    task_prompt = task.prompt[:50] + "..." if len(task.prompt) > 50 else task.prompt

    with hud.trace(name=task_prompt):
        agent = build_agent(
            agent_type,
            model=model,
            allowed_tools=allowed_tools,
            verbose=verbose,
        )
        hud_console.info(task.prompt)
        result = await agent.run(task, max_steps=max_steps)
        hud_console.success(f"Reward: {result.reward}")


async def run_full_dataset(
    source: str,
    *,
    agent_type: Literal["claude", "openai"] = "claude",
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_concurrent: int = 50,
    max_steps: int = 10,
    parallel: bool = False,
    max_workers: int | None = None,
    max_concurrent_per_worker: int = 25,
    verbose: bool = False,
) -> list[Any]:
    """Run evaluation across the entire dataset.

    Uses either asyncio-based run_dataset or process-based parallel execution
    depending on the parallel flag."""

    # Import run_dataset lazily
    try:
        from hud.datasets import run_dataset, run_dataset_parallel, run_dataset_parallel_manual
    except ImportError as e:
        hud_console.error(
            "Dataset dependencies are not installed. "
            "Please install with: pip install 'hud-python[[agent]]'"
        )
        raise typer.Exit(1) from e

    # Check if source is a JSON file with list of tasks
    path = Path(source)
    dataset_or_tasks = source
    dataset_name = source.split("/")[-1]

    if path.exists() and path.suffix == ".json":
        with open(path) as f:  # noqa: ASYNC230
            json_data = json.load(f)

        if isinstance(json_data, list):
            dataset_or_tasks = json_data
            dataset_name = f"JSON Dataset: {path.name}"
            hud_console.info(f"Found {len(json_data)} tasks in JSON file")
        else:
            hud_console.error("JSON file must contain a list of tasks when using --full flag")
            raise typer.Exit(1)

    # Build agent class + config for run_dataset
    if agent_type == "openai":
        try:
            from hud.agents import OperatorAgent

            agent_class = OperatorAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        agent_config: dict[str, Any] = {"verbose": verbose}
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools

    else:
        try:
            from hud.agents import ClaudeAgent

            agent_class = ClaudeAgent
        except ImportError as e:
            hud_console.error(
                "Claude agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        agent_config = {
            "model": model or "claude-sonnet-4-20250514",
            "verbose": verbose,
        }
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools

    if parallel:
        hud_console.info(
            f"🚀 Running PARALLEL evaluation (workers: {max_workers or 'auto'}, max_concurrent: {max_concurrent})…"  # noqa: E501
        )
        if max_workers is None:
            # Use auto-optimization (now the default run_dataset_parallel)
            return await run_dataset_parallel(
                name=f"Evaluation {dataset_name}",
                dataset=dataset_or_tasks,
                agent_class=agent_class,
                agent_config=agent_config,
                max_concurrent=max_concurrent,
                metadata={"dataset": source, "parallel": True},
                max_steps=max_steps,
                auto_respond=True,
            )
        else:
            # Use manual configuration
            return await run_dataset_parallel_manual(
                name=f"Evaluation {dataset_name}",
                dataset=dataset_or_tasks,
                agent_class=agent_class,
                agent_config=agent_config,
                max_workers=max_workers,
                max_concurrent_per_worker=max_concurrent_per_worker,
                max_concurrent=max_concurrent,
                metadata={"dataset": source, "parallel": True},
                max_steps=max_steps,
                auto_respond=True,
            )
    else:
        hud_console.info(f"🚀 Running evaluation (max_concurrent: {max_concurrent})…")
        return await run_dataset(
            name=f"Evaluation {dataset_name}",
            dataset=dataset_or_tasks,
            agent_class=agent_class,
            agent_config=agent_config,
            max_concurrent=max_concurrent,
            metadata={"dataset": source},
            max_steps=max_steps,
        )


def eval_command(
    source: str = typer.Argument(
        ...,
        help="HuggingFace dataset identifier (e.g. 'hud-evals/SheetBench-50'), single task JSON file, or JSON file with list of tasks",  # noqa: E501
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run the entire dataset (omit for single-task debug mode)",
    ),
    agent: Literal["claude", "openai"] = typer.Option(
        "claude",
        "--agent",
        help="Agent backend to use",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model name for the chosen agent",
    ),
    allowed_tools: str | None = typer.Option(
        None,
        "--allowed-tools",
        help="Comma-separated list of allowed tools",
    ),
    max_concurrent: int = typer.Option(
        50,
        "--max-concurrent",
        help="Concurrency level for asyncio mode (ignored in parallel mode)",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        help="Maximum steps per task (default: 10 for single, 50 for full)",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Use process-based parallel execution for large datasets (100+ tasks)",
    ),
    max_workers: int | None = typer.Option(
        None,
        "--max-workers",
        help="Number of worker processes for parallel mode (auto-optimized if not set)",
    ),
    max_concurrent_per_worker: int = typer.Option(
        20,
        "--max-concurrent-per-worker",
        help="Maximum concurrent tasks per worker in parallel mode",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output from the agent",
    ),
) -> None:
    """🚀 Run evaluation on datasets or individual tasks with agents.

    Examples:
        # Evaluate a single task from SheetBench
        hud eval hud-evals/SheetBench-50

        # Evaluate the FULL SheetBench dataset with Claude (asyncio mode)
        hud eval hud-evals/SheetBench-50 --full --agent claude

        # Run large dataset with PARALLEL execution (auto-optimized)
        hud eval hud-evals/OSWorld-Verified-XLang --full --parallel

        # Parallel mode with manual configuration (16 workers, 25 tasks each)
        hud eval hud-evals/OSWorld-Verified-XLang --full --parallel --max-workers 16

        # Limit total concurrent tasks to prevent rate limits
        hud eval hud-evals/SheetBench-50 --full --parallel --max-concurrent 20

        # Run a single task from a JSON file
        hud eval task.json

        # Run multiple tasks from a JSON file with parallel execution
        hud eval tasks.json --full --parallel

        # Run with OpenAI Operator agent
        hud eval hud-evals/OSWorld-Gold-Beta --agent openai

        # Run with verbose output for debugging
        hud eval task.json --verbose
    """
    from hud.settings import settings

    # Check for required API keys
    if agent == "claude":
        if not settings.anthropic_api_key:
            hud_console.error("ANTHROPIC_API_KEY is required for Claude agent")
            hud_console.info(
                "Set it in your environment or .env file: ANTHROPIC_API_KEY=your-key-here"
            )
            raise typer.Exit(1)
    elif agent == "openai" and not settings.openai_api_key:
        hud_console.error("OPENAI_API_KEY is required for OpenAI agent")
        hud_console.info("Set it in your environment or .env file: OPENAI_API_KEY=your-key-here")
        raise typer.Exit(1)

    # Check for HUD_API_KEY if using HUD services
    if not settings.api_key:
        hud_console.warning("HUD_API_KEY not set. Some features may be limited.")
        hud_console.info("Get your API key at: https://app.hud.so")

    # Parse allowed tools
    allowed_tools_list = (
        [t.strip() for t in allowed_tools.split(",") if t.strip()] if allowed_tools else None
    )

    # Set default max_steps if not provided
    if max_steps is None:
        max_steps = 50 if full else 10

    # Run evaluation
    if full:
        asyncio.run(
            run_full_dataset(
                source,
                agent_type=agent,
                model=model,
                allowed_tools=allowed_tools_list,
                max_concurrent=max_concurrent,
                max_steps=max_steps,
                parallel=parallel,
                max_workers=max_workers,
                max_concurrent_per_worker=max_concurrent_per_worker,
                verbose=verbose,
            )
        )
    else:
        asyncio.run(
            run_single_task(
                source,
                agent_type=agent,
                model=model,
                allowed_tools=allowed_tools_list,
                max_steps=max_steps,
                verbose=verbose,
            )
        )
