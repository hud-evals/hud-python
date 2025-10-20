"""HUD evaluation command for running tasks and datasets."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

import hud
from hud.cli.utils.env_check import ensure_built, find_environment_dir
from hud.settings import settings
from hud.types import AgentType
from hud.utils.group_eval import display_group_statistics, run_tasks_grouped
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from hud.types import Task
logger = logging.getLogger(__name__)
hud_console = HUDConsole()


def get_available_models() -> list[dict[str, str | None]]:
    """Fetch available models from the HUD API (only ready models).

    Returns:
        List of dicts with 'name', 'vllm_url', and 'base_model' keys
    """
    try:
        from hud.cli.rl import rl_api

        hud_console.info("Fetching your models from https://hud.so/models")
        models = rl_api.list_models()

        # Filter for ready models only and sort by recency
        ready_models = [m for m in models if m.status == "ready"]
        ready_models.sort(key=lambda m: m.created_at or "", reverse=True)

        # Count other statuses for informational purposes
        training_count = sum(1 for m in models if m.status == "training")
        # other_count = len(models) - len(ready_models) - training_count

        if ready_models:
            hud_console.success(f"Found {len(ready_models)} ready models:")
            for model in ready_models:
                vllm_status = " (vLLM deployed)" if model.vllm_url else ""
                hud_console.info(f"  ✅ {model.name}{vllm_status}")

            if training_count > 0:
                hud_console.info(f"\n({training_count} models currently training)")

            return [
                {"name": model.name, "vllm_url": model.vllm_url, "base_model": model.base_model}
                for model in ready_models
            ]
        else:
            if training_count > 0:
                hud_console.warning(
                    f"No ready models found. You have {training_count} models currently training."
                )
            else:
                hud_console.warning("No models found in your account.")
            return []
    except Exception as e:
        hud_console.debug(f"Error fetching models: {e}")
        # Don't show the error to the user, just proceed without HUD models
        return []


def _build_vllm_config(
    vllm_base_url: str | None,
    model: str | None,
    allowed_tools: list[str] | None,
    verbose: bool,
) -> dict[str, Any]:
    """Build configuration for vLLM agent.

    Args:
        vllm_base_url: Optional base URL for vLLM server
        model: Model name to use
        allowed_tools: Optional list of allowed tools
        verbose: Enable verbose output

    Returns:
        Dictionary with agent configuration
    """
    # Determine base URL and API key
    if vllm_base_url is not None:
        base_url = vllm_base_url
        api_key = settings.api_key if base_url.startswith(settings.hud_rl_url) else "token-abc123"
        hud_console.info(f"Using vLLM server at {base_url}")
    else:
        base_url = "http://localhost:8000/v1"
        api_key = "token-abc123"

    config: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
        "model_name": model or "served-model",
        "verbose": verbose,
        "completion_kwargs": {
            "temperature": 0.7,
            "max_tokens": 2048,
            "tool_choice": "auto",
        },
    }

    if allowed_tools:
        config["allowed_tools"] = allowed_tools

    return config


def build_agent(
    agent_type: AgentType,
    *,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    verbose: bool = False,
    vllm_base_url: str | None = None,
) -> Any:
    """Create and return the requested agent type."""

    # Import agents lazily to avoid dependency issues
    if agent_type == AgentType.INTEGRATION_TEST:
        from hud.agents.misc.integration_test_agent import IntegrationTestRunner

        return IntegrationTestRunner(verbose=verbose)
    elif agent_type == AgentType.VLLM:
        # Create a generic OpenAI agent for vLLM server
        try:
            from hud.agents.openai_chat_generic import GenericOpenAIChatAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        # Use the shared config builder
        config = _build_vllm_config(
            vllm_base_url=vllm_base_url,
            model=model,
            allowed_tools=allowed_tools,
            verbose=verbose,
        )
        return GenericOpenAIChatAgent(**config)

    elif agent_type == AgentType.OPENAI:
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

    elif agent_type == AgentType.LITELLM:
        try:
            from hud.agents.lite_llm import LiteAgent
        except ImportError as e:
            hud_console.error(
                "LiteLLM agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        return LiteAgent(
            model_name=model or "gpt-4o-mini",
            allowed_tools=allowed_tools,
            verbose=verbose,
        )

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
    agent_type: AgentType = AgentType.CLAUDE,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_steps: int = 10,
    verbose: bool = False,
    vllm_base_url: str | None = None,
    group_size: int = 1,
) -> None:
    """Load one task and execute it, or detect if JSON contains a list and run as dataset."""

    # Provide early feedback to user
    hud_console.info("🔧 Initializing evaluation...")
    # Import Task and run_dataset lazily
    try:
        from hud.utils.tasks import load_tasks
    except ImportError as e:
        hud_console.error(
            "Dataset dependencies are not installed. "
            "Please install with: pip install 'hud-python\u27e6agent\u27e7'"
        )
        raise typer.Exit(1) from e

    path = Path(source)
    if path.exists() and (path.suffix in [".json", ".jsonl"]):
        hud_console.info("📊 Loading task file…")
        tasks: list[Task] = load_tasks(str(path))  # type: ignore[assignment]

        # If tasks reference a local environment (nearby), ensure it's built/up-to-date.
        try:
            env_dir = find_environment_dir(path)
            if env_dir is not None:
                # Non-interactive for eval; warn but don't block
                ensure_built(env_dir, interactive=False)
        except Exception as e:
            hud_console.debug(f"Eval preflight env check skipped: {e}")

        # Single task - use the first (and only) task
        task = tasks[0]
        hud_console.info("Found 1 task, running as single task…")

    else:
        # Load from HuggingFace dataset or non-file source
        hud_console.info(f"📊 Loading tasks from: {source}…")
        tasks: list[Task] = load_tasks(source)  # type: ignore[assignment]

        if not tasks:
            hud_console.error(f"No tasks found in: {source}")
            raise typer.Exit(1)

        # Single task - use the first task
        task = tasks[0]
        hud_console.info(
            "Using first task from dataset (run with --full to run the entire dataset)..."
        )

    task_prompt = task.prompt[:50] + "..." if len(task.prompt) > 50 else task.prompt

    # Use grouped evaluation if group_size > 1
    agent_config: dict[str, Any] = {}
    if agent_type == AgentType.INTEGRATION_TEST:
        from hud.agents.misc.integration_test_agent import IntegrationTestRunner

        agent_class = IntegrationTestRunner
        agent_config = {"verbose": verbose}
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools
    elif agent_type == AgentType.VLLM:
        # Special handling for vLLM
        from hud.agents.openai_chat_generic import GenericOpenAIChatAgent

        agent_class = GenericOpenAIChatAgent

        # Use the shared config builder
        agent_config = _build_vllm_config(
            vllm_base_url=vllm_base_url,
            model=model,
            allowed_tools=allowed_tools,
            verbose=verbose,
        )
    elif agent_type == AgentType.OPENAI:
        from hud.agents import OperatorAgent

        agent_class = OperatorAgent
        agent_config = {"verbose": verbose}
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools
    elif agent_type == AgentType.LITELLM:
        from hud.agents.lite_llm import LiteAgent

        agent_class = LiteAgent
        agent_config = {
            "model_name": model or "gpt-4o-mini",
            "verbose": verbose,
        }
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools
    elif agent_type == AgentType.CLAUDE:
        from hud.agents import ClaudeAgent

        agent_class = ClaudeAgent
        agent_config = {
            "model": model or "claude-sonnet-4-20250514",
            "verbose": verbose,
            "validate_api_key": False,
        }
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

    if group_size > 1:
        hud_console.info(f"🔄 Running task with group_size={group_size}")
        # Run with grouping
        stats = await run_tasks_grouped(
            tasks=[task],
            agent_class=agent_class,
            agent_config=agent_config,
            group_size=group_size,
            max_parallel_episodes=48,  # Same as RL default
            max_steps=max_steps,
            verbose=verbose,
        )
        display_group_statistics(stats, show_details=True)
    else:
        # Enable agent step logging for single task mode
        logging.getLogger("hud.agents").setLevel(logging.INFO)
        logging.getLogger("hud.agents.base").setLevel(logging.INFO)

        with hud.trace(name=task_prompt):
            agent = build_agent(
                agent_type,
                model=model,
                allowed_tools=allowed_tools,
                verbose=verbose,
                vllm_base_url=vllm_base_url,
            )
            hud_console.info(task.prompt)
            result = await agent.run(task, max_steps=max_steps)
            hud_console.success(f"Reward: {result.reward}")


async def run_full_dataset(
    source: str,
    *,
    agent_type: AgentType = AgentType.CLAUDE,
    model: str | None = None,
    allowed_tools: list[str] | None = None,
    max_concurrent: int = 30,
    max_steps: int = 10,
    verbose: bool = False,
    vllm_base_url: str | None = None,
    group_size: int = 1,
) -> list[Any]:
    """Run evaluation across the entire dataset using asyncio-based concurrency."""

    # Provide early feedback to user
    hud_console.info("🔧 Initializing evaluation...")

    # Import run_dataset lazily
    try:
        from hud.datasets import run_dataset
        from hud.utils.tasks import load_tasks
    except ImportError as e:
        hud_console.error(
            "Dataset dependencies are not installed. "
            "Please install with: pip install 'hud-python[agent]'"
        )
        raise typer.Exit(1) from e

    # Load tasks using unified loader
    hud_console.info(f"📊 Loading tasks from: {source}…")
    tasks: list[Task] = load_tasks(source)  # type: ignore[assignment]

    if len(tasks) == 0:
        hud_console.error(f"No tasks found in: {source}")
        raise typer.Exit(1)

    # Convert Task objects to dicts for dataset runners
    dataset_or_tasks = [task.model_dump() for task in tasks]

    # Determine dataset name
    path = Path(source)
    dataset_name = f"Dataset: {path.name}" if path.exists() else source.split("/")[-1]

    # Build agent class + config for run_dataset
    agent_config: dict[str, Any]
    if agent_type == AgentType.INTEGRATION_TEST:  # --integration-test mode
        from hud.agents.misc.integration_test_agent import IntegrationTestRunner

        agent_class = IntegrationTestRunner
        agent_config = {"verbose": verbose}
    elif agent_type == AgentType.VLLM:
        try:
            from hud.agents.openai_chat_generic import GenericOpenAIChatAgent

            agent_class = GenericOpenAIChatAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        # Use the shared config builder
        agent_config = _build_vllm_config(
            vllm_base_url=vllm_base_url,
            model=model,
            allowed_tools=allowed_tools,
            verbose=verbose,
        )
    elif agent_type == AgentType.OPENAI:
        try:
            from hud.agents import OperatorAgent

            agent_class = OperatorAgent
        except ImportError as e:
            hud_console.error(
                "OpenAI agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        agent_config = {"verbose": verbose, "validate_api_key": False}
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools

    elif agent_type == AgentType.LITELLM:
        try:
            from hud.agents.lite_llm import LiteAgent

            agent_class = LiteAgent
        except ImportError as e:
            hud_console.error(
                "LiteLLM agent dependencies are not installed. "
                "Please install with: pip install 'hud-python[agent]'"
            )
            raise typer.Exit(1) from e

        agent_config = {
            "model_name": model or "gpt-4o-mini",
            "verbose": verbose,
        }
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
            "validate_api_key": False,
        }
        if allowed_tools:
            agent_config["allowed_tools"] = allowed_tools

    # Use grouped evaluation if group_size > 1
    if group_size > 1:
        hud_console.info(f"🔄 Running dataset with group_size={group_size}")

        # Run with job tracking
        with hud.job(
            name=f"Evaluation {dataset_name} (group_size={group_size})",
            metadata={
                "dataset": source,
                "group_size": group_size,
                "tasks": len(dataset_or_tasks),
                "total_episodes": len(dataset_or_tasks) * group_size,
            },
        ) as job:
            # Convert dicts to Task objects if needed
            from hud.datasets import Task

            tasks = []
            for item in dataset_or_tasks:
                if isinstance(item, dict):
                    tasks.append(Task(**item))
                else:
                    tasks.append(item)

            stats = await run_tasks_grouped(
                tasks=tasks,
                agent_class=agent_class,
                agent_config=agent_config,
                group_size=group_size,
                max_parallel_episodes=max_concurrent,
                max_steps=max_steps,
                verbose=verbose,
                job_id=job.id,
            )

        # Display results
        display_group_statistics(stats, show_details=len(stats) <= 50)

        # Return stats for consistency with other modes
        return stats

    # Run evaluation with asyncio-based concurrency
    hud_console.info(f"🚀 Running evaluation (max_concurrent: {max_concurrent})…")
    return await run_dataset(
        name=f"Evaluation {dataset_name}",
        dataset=dataset_or_tasks,
        agent_class=agent_class,
        agent_config=agent_config,
        max_concurrent=max_concurrent,
        metadata={"dataset": source},
        max_steps=max_steps,
        auto_respond=True,
    )


def eval_command(
    source: str = typer.Argument(
        ...,
        help="HuggingFace dataset identifier (e.g. 'hud-evals/SheetBench-50'), JSON file (array of tasks), or JSONL file (one task per line)",  # noqa: E501
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run the entire dataset (omit for single-task debug mode)",
    ),
    agent: AgentType = typer.Option(  # noqa: B008
        AgentType.CLAUDE,
        "--agent",
        help="Agent backend to use (claude, openai, vllm for local server, or litellm)",
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
        30,
        "--max-concurrent",
        help=(
            "Maximum concurrent tasks (1-200 recommended, prevents rate limits "
            "and resource exhaustion)"
        ),
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        help="Maximum steps per task (default: 10 for single, 50 for full)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output from the agent",
    ),
    very_verbose: bool = typer.Option(
        False,
        "--very-verbose",
        "-vv",
        help="Enable debug-level logs for maximum visibility",
    ),
    vllm_base_url: str | None = typer.Option(
        None,
        "--vllm-base-url",
        help="Base URL for vLLM server (when using --agent vllm)",
    ),
    group_size: int = typer.Option(
        1,
        "--group-size",
        help="Number of times to run each task (similar to RL training)",
    ),
    integration_test: bool = typer.Option(
        False,
        "--integration-test",
        help=(
            "Run integration_test_tool tool, where problem is setup, "
            "actions are applied, and evaluation is performed, without "
            "spinning up an agent"
        ),
    ),
) -> None:
    """🚀 Run evaluation on datasets or individual tasks with agents.

    Examples:
        # Evaluate a single task from SheetBench
        hud eval hud-evals/SheetBench-50

        # Evaluate the FULL SheetBench dataset with Claude
        hud eval hud-evals/SheetBench-50 --full --agent claude

        # Run with higher concurrency for faster evaluation
        hud eval hud-evals/OSWorld-Verified-Gold --full --max-concurrent 100

        # Limit concurrent tasks to prevent rate limits
        hud eval hud-evals/SheetBench-50 --full --max-concurrent 20

        # Run a single task from a JSON file
        hud eval task.json

        # Run multiple tasks from a JSON file
        hud eval tasks.json --full

        # Run with OpenAI Operator agent
        hud eval hud-evals/OSWorld-Gold-Beta --agent openai

        # Use local vLLM server (default: localhost:8000)
        hud eval task.json --agent vllm --model Qwen/Qwen2.5-VL-3B-Instruct

        # Use custom vLLM server URL
        hud eval task.json --agent vllm --vllm-base-url http://192.168.1.100:8000/v1

        # Run with verbose output for debugging
        hud eval task.json --verbose
    """
    # Always configure basic logging so agent steps can be logged
    # Set to INFO by default for consistency with run_evaluation.py
    if very_verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        logging.getLogger("hud.agents").setLevel(logging.DEBUG)
        logging.getLogger("hud.agents.base").setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger("hud.agents").setLevel(logging.INFO)
        logging.getLogger("hud.agents.base").setLevel(logging.INFO)

    # We pass integration_test as the agent_type
    if integration_test:
        agent = AgentType.INTEGRATION_TEST

    # Check for required API keys
    if agent == AgentType.CLAUDE:
        if not settings.anthropic_api_key:
            hud_console.error("ANTHROPIC_API_KEY is required for Claude agent")
            hud_console.info(
                "Set it in your environment or run: hud set ANTHROPIC_API_KEY=your-key-here"
            )
            raise typer.Exit(1)
    elif agent == AgentType.OPENAI and not settings.openai_api_key:
        hud_console.error("OPENAI_API_KEY is required for OpenAI agent")
        hud_console.info("Set it in your environment or run: hud set OPENAI_API_KEY=your-key-here")
        raise typer.Exit(1)
    elif agent == AgentType.VLLM:
        if model:
            hud_console.info(f"Using vLLM with model: {model}")
        else:
            hud_console.error("Model name is required for vLLM agent, specify with --model")
            raise typer.Exit(1)

    # Check for HUD_API_KEY if using HUD services
    if not settings.api_key:
        hud_console.warning("HUD_API_KEY not set. Some features may be limited.")
        hud_console.info("Get your API key at: https://hud.so")
        hud_console.info("Set it in your environment or run: hud set HUD_API_KEY=your-key-here")

    # Parse allowed tools
    allowed_tools_list = (
        [t.strip() for t in allowed_tools.split(",") if t.strip()] if allowed_tools else None
    )

    # Set default max_steps if not provided
    if max_steps is None:
        max_steps = 50 if full else 10

    # Run evaluation
    if full:
        import time

        start_time = time.time()

        results = asyncio.run(
            run_full_dataset(
                source,
                agent_type=agent,
                model=model,
                allowed_tools=allowed_tools_list,
                max_concurrent=max_concurrent,
                max_steps=max_steps,
                verbose=very_verbose or verbose,
                vllm_base_url=vllm_base_url,
                group_size=group_size,
            )
        )

        elapsed = time.time() - start_time

        # Print statistics (only for non-grouped mode)
        if group_size == 1 and results:
            hud_console.info("\n" + "=" * 50)
            hud_console.success("📊 Evaluation Complete!")
            hud_console.info("=" * 50)
            hud_console.info(f"Total tasks: {len(results)}")
            hud_console.info(f"Time elapsed: {elapsed:.2f} seconds")
            hud_console.info(f"Throughput: {len(results) / elapsed:.2f} tasks/second")
            hud_console.info(f"Execution mode: ASYNCIO (max_concurrent: {max_concurrent})")

            # Count successes
            successful = sum(1 for r in results if getattr(r, "reward", 0) > 0.7)
            success_rate = 100 * successful / len(results)
            hud_console.info(f"Successful tasks: {successful}/{len(results)} ({success_rate:.1f}%)")
            hud_console.info("=" * 50)
    else:
        asyncio.run(
            run_single_task(
                source,
                agent_type=agent,
                model=model,
                allowed_tools=allowed_tools_list,
                max_steps=max_steps,
                verbose=very_verbose or verbose,
                vllm_base_url=vllm_base_url,
                group_size=group_size,
            )
        )
