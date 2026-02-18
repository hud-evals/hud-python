"""CLI commands for collecting rollout trajectories."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer

from hud.rl.collector import collect_rollouts, write_rollouts_jsonl
from hud.types import AgentType
from hud.utils.hud_console import HUDConsole

rollout_app = typer.Typer(help="Collect rollout trajectories for RL/RFT workflows.")
hud_console = HUDConsole()


def _resolve_agent(
    *,
    agent: AgentType,
    model: str | None,
    allowed_tools: list[str] | None,
    disallowed_tools: list[str] | None,
    verbose: bool,
) -> tuple[AgentType, dict[str, Any]]:
    config: dict[str, Any] = {"verbose": verbose, "validate_api_key": False}
    if model is not None and agent != AgentType.INTEGRATION_TEST:
        config["model"] = model
    if allowed_tools:
        config["allowed_tools"] = allowed_tools
    if disallowed_tools:
        config["disallowed_tools"] = disallowed_tools
    return agent, config


@rollout_app.command("collect")
def collect_command(
    source: str | None = typer.Argument(
        None,
        help=(
            "HuggingFace dataset name (e.g. hud-evals/SheetBench-50) or path to "
            "a local JSON/JSONL tasks file."
        ),
    ),
    output: Path = typer.Option(  # noqa: B008
        Path("rollouts.jsonl"),
        "--output",
        "-o",
        help="Output JSONL file for collected trajectories.",
    ),
    agent: AgentType = typer.Option(  # noqa: B008
        AgentType.CLAUDE,
        "--agent",
        help="Agent backend to use for rollout collection.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model name for the selected agent backend.",
    ),
    allowed_tools: str | None = typer.Option(
        None,
        "--allowed-tools",
        help="Comma-separated list of allowed tools.",
    ),
    disallowed_tools: str | None = typer.Option(
        None,
        "--disallowed-tools",
        help="Comma-separated list of disallowed tools.",
    ),
    max_concurrent: int = typer.Option(
        30,
        "--max-concurrent",
        min=1,
        help="Maximum number of concurrent tasks.",
    ),
    max_steps: int = typer.Option(
        50,
        "--max-steps",
        min=1,
        help="Maximum steps per rollout.",
    ),
    group_size: int = typer.Option(
        1,
        "--group-size",
        min=1,
        help="Number of rollouts to collect per task.",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        help="Dataset split when source is a HuggingFace dataset.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose agent logs.",
    ),
) -> None:
    """Collect and export rollout trajectories."""
    if source is None:
        from .utils.tasks import find_tasks_file

        try:
            source = find_tasks_file(None, msg="Select a tasks file to collect rollouts from")
            hud_console.success(f"Selected: {source}")
        except (FileNotFoundError, Exception):
            hud_console.error(
                "No source provided and no task/eval JSON files found in current directory"
            )
            raise typer.Exit(1) from None

    allowed_tools_list = (
        [tool.strip() for tool in allowed_tools.split(",") if tool.strip()]
        if allowed_tools
        else None
    )
    disallowed_tools_list = (
        [tool.strip() for tool in disallowed_tools.split(",") if tool.strip()]
        if disallowed_tools
        else None
    )
    agent_type, agent_params = _resolve_agent(
        agent=agent,
        model=model,
        allowed_tools=allowed_tools_list,
        disallowed_tools=disallowed_tools_list,
        verbose=verbose,
    )

    run_name = f"Rollout Collection: {Path(source).name if Path(source).exists() else source}"
    records = asyncio.run(
        collect_rollouts(
            name=run_name,
            source=source,
            agent_type=agent_type,
            agent_params=agent_params,
            max_concurrent=max_concurrent,
            max_steps=max_steps,
            split=split,
            group_size=group_size,
            metadata={"source": source, "group_size": group_size},
            auto_respond=True,
        )
    )

    if not records:
        hud_console.warning("No rollouts were collected.")
        return

    output_path = write_rollouts_jsonl(records, output)
    hud_console.success(f"Collected {len(records)} rollouts")
    hud_console.info(f"Saved to: {output_path}")
