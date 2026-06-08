"""HUD evaluation command for running tasks and datasets.

Config Override Order: CLI arguments > .hud_eval.toml > defaults
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, cast

import questionary
import typer

from hud.cli.eval_config import EvalConfig
from hud.settings import settings
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def _build_agent(cfg: EvalConfig) -> Any:
    """Construct a new-flow agent (``agent(run)``) from the eval config.

    New agents are config-based: ``AgentType.cls(config=AgentType.config_cls(...))``.
    Eval-config kwargs are mapped onto the agent's config (unknown keys ignored).
    """
    if cfg.agent_type is None:
        raise ValueError("agent_type must be set")
    agent_kwargs = cfg.get_agent_kwargs()
    if cfg.auto_respond:
        agent_kwargs["auto_respond"] = True
    config = cfg.agent_type.config_cls.model_validate(agent_kwargs)
    # cls/config_cls are matched unions; the pairing is correct by construction.
    return cast("Any", cfg.agent_type.cls)(config=config)


async def _run_evaluation(cfg: EvalConfig) -> tuple[list[Any], list[Any]]:
    """Run evaluation on the new Env/Variant/Taskset/Run flow.

    Loads runnable ``Variant``s from a Python source (a ``.py`` file or directory
    defining a :class:`hud.env.Env` with ``@env.task``), builds a ``Taskset``, and
    runs the agent. Legacy JSON/JSONL files, API tasksets, and remote submission
    are not supported on this flow yet.
    """
    from hud.eval.source import load_variants

    if cfg.source is None or cfg.agent_type is None:
        raise ValueError("source and agent_type must be set")

    if cfg.remote:
        hud_console.error(
            "Remote execution is not supported on the new eval flow yet. "
            "Run locally against a Python Env source or a JSON taskset."
        )
        raise typer.Exit(1)

    path = Path(cfg.source)
    if not path.exists():
        hud_console.error(
            "`hud eval` runs the new Env/Variant flow. Pass a Python source "
            "(a .py file or directory defining a `hud.env.Env` with `@env.task`) or a "
            f"JSON/JSONL taskset. API tasksets are not supported yet (got: {cfg.source})."
        )
        raise typer.Exit(1)

    hud_console.info(f"Loading variants from: {cfg.source}")
    try:
        if path.suffix not in {".json", ".jsonl", ".py"} and not path.is_dir():
            hud_console.error(
                f"Unsupported source type: {path.suffix} (expected .py, .json, .jsonl, or a dir)."
            )
            raise typer.Exit(1)
        variants = load_variants(path)
    except typer.Exit:
        raise
    except Exception as e:
        hud_console.error(f"Failed to load variants from {cfg.source}: {e}")
        raise typer.Exit(1) from e

    if not variants:
        hud_console.error(
            f"No runnable Variants found in {cfg.source}. Define a `hud.env.Env` with "
            "`@env.task` and expose Variants (e.g. `t = my_task(arg=...)`). "
            "(Legacy env+scenario Tasks are not supported on the new flow.)"
        )
        raise typer.Exit(1)

    # Filter by task name or positional index, or default to the first variant.
    if cfg.task_ids:
        selector = set(cfg.task_ids)
        filtered = [
            v
            for i, v in enumerate(variants)
            if getattr(v, "task", None) in selector or str(i) in selector
        ]
        if not filtered:
            hud_console.error(f"No variants matching: {', '.join(cfg.task_ids)}")
            raise typer.Exit(1)
        hud_console.info(f"Filtered to {len(filtered)} variant(s)")
        variants = filtered
    elif not cfg.all:
        variants = [variants[0]]
        hud_console.info("Using first variant (run with --full or --task-ids for more)…")

    hud_console.info(f"Loaded {len(variants)} variant(s)")

    if len(variants) == 1 and cfg.group_size == 1:
        logging.getLogger("hud.agents").setLevel(logging.INFO)
    else:
        hud_console.info(
            f"🚀 Running evaluation (max_concurrent: {cfg.max_concurrent}, "
            f"group_size: {cfg.group_size})…"
        )

    from hud.eval import Taskset

    agent = _build_agent(cfg)
    runs = await Taskset(variants).run(
        agent,
        group=cfg.group_size,
        max_concurrent=cfg.max_concurrent,
        max_steps=cfg.max_steps,
    )

    job_id = runs[0].job_id if runs else None
    if job_id and settings.telemetry_enabled and settings.api_key:
        hud_console.info(f"🔗 https://hud.ai/jobs/{job_id}")

    return runs, variants


# =============================================================================
# CLI command
# =============================================================================


def eval_command(
    source: str | None = typer.Argument(None, help="Taskset slug or task JSON file"),
    agent: str | None = typer.Argument(
        None,
        help="Agent: claude, openai, gemini, openai_compatible",
    ),
    all: bool = typer.Option(False, "--all", help="Run all problems instead of just 1"),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run the entire dataset. Shortcut for --all --auto-respond  --max-steps 100",
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name"),
    config: list[str] | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Agent config: key=value"
    ),
    from_json: Path | None = typer.Option(  # noqa: B008
        None,
        "--from-json",
        help="Load full eval configuration from a JSON file (e.g. exported from a HUD job).",
    ),
    # Eval settings
    max_concurrent: int | None = typer.Option(
        None, "--max-concurrent", help="Max concurrent tasks"
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Max steps per task"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    very_verbose: bool = typer.Option(False, "--very-verbose", "-vv", help="Debug logs"),
    auto_respond: bool = typer.Option(
        False,
        "--auto-respond",
        help="Automatically prompt the agent to continue if it does not respond with a tool call",
    ),
    group_size: int | None = typer.Option(None, "--group-size", help="Runs per task"),
    task_ids: str | None = typer.Option(
        None,
        "--task-ids",
        help="Comma-separated task slugs (or 0-based indices) to run",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    remote: bool = typer.Option(
        False, "--remote", help="Submit tasks to platform for remote execution"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress opening browser for eval links"
    ),
    gateway: bool = typer.Option(
        False, "--gateway", "-g", help="Route LLM API calls through HUD Gateway"
    ),
    taskset: str | None = typer.Option(
        None, "--taskset", "-t", help="Taskset name to associate job with"
    ),
) -> None:
    """🚀 Run evaluation on datasets or individual tasks with agents.

    Examples:
        hud eval tasks.json claude
        hud eval "My Tasks" claude --full              # Load from platform taskset
        hud eval tasks.json claude --taskset "My Tasks" # Associate file tasks with taskset
        hud eval tasks.json claude --config max_tokens=32768
        hud eval tasks.json claude --full --remote     # Remote execution
        hud eval tasks.json claude --gateway           # Route LLM calls through HUD Gateway
    """
    hud_console.info("🔧 Initializing evaluation...")

    # Load config (TOML by default), optionally override with a JSON config, then merge CLI args
    if from_json is not None:
        try:
            cfg = EvalConfig.model_validate_json(from_json.read_text(encoding="utf-8"))
        except Exception as e:
            hud_console.error(f"Failed to load JSON config from {from_json}: {e}")
            raise typer.Exit(1) from None
    else:
        cfg = EvalConfig.load()

    cfg = cfg.merge_cli(
        source=source,
        agent=agent,
        model=model,
        all=all,
        full=full,
        max_concurrent=max_concurrent,
        max_steps=max_steps,
        task_ids=task_ids,
        verbose=verbose,
        very_verbose=very_verbose,
        auto_respond=auto_respond,
        group_size=group_size,
        config=config,
        remote=remote,
        quiet=quiet,
        gateway=gateway,
        taskset=taskset,
    )

    # Find source if not provided
    if cfg.source is None:
        try:
            from hud.cli.utils.tasks import find_tasks_file

            cfg = cfg.model_copy(
                update={"source": find_tasks_file(None, msg="Select a tasks file")}
            )
            hud_console.success(f"Selected: {cfg.source}")
        except Exception:
            hud_console.error("No source provided and no task files found")
            raise typer.Exit(1) from None

    # Resolve agent interactively if needed
    cfg = cfg.resolve_agent_interactive()

    # Configure logging
    if cfg.very_verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(message)s")
        logging.getLogger("hud.agents").setLevel(logging.DEBUG)
        # Suppress noisy HTTP client logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    elif cfg.verbose:
        logging.getLogger("hud.agents").setLevel(logging.INFO)

    # Validate API keys
    cfg.validate_api_keys()

    # Fail fast before asking the user to confirm settings we cannot honor.
    if cfg.remote:
        hud_console.error(
            "Remote execution is not supported on the new eval flow yet. "
            "Run locally against a Python Env source or a JSON taskset."
        )
        raise typer.Exit(1)

    # Display and confirm
    cfg.display()

    if not yes and not questionary.confirm("Proceed?", default=True, qmark="").ask():
        hud_console.info("Cancelled.")
        raise typer.Exit(1)

    # Run
    start_time = time.time()
    try:
        results, _tasks = asyncio.run(_run_evaluation(cfg))
    except ValueError as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from None
    elapsed = time.time() - start_time

    if results:
        from hud.cli.utils.display import display_runs

        display_runs(results, name=cfg.source or "", elapsed=elapsed)
