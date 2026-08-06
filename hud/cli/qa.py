"""Discover, run, and inspect resource-scoped platform QA agents."""

from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import Any, cast

import typer

from hud.cli.utils.api import require_api_key
from hud.utils.exceptions import HudTimeoutError
from hud.utils.platform import PlatformClient

_POLL_INTERVAL_SECONDS = 2.0
_TERMINAL_STATUSES = {"completed", "error"}


class ResourceSubjectType(StrEnum):
    environment = "environment"
    taskset = "taskset"
    task = "task"


qa_app = typer.Typer(
    name="qa",
    help="Discover, run, and inspect platform QA agents.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _render_results(results: list[dict[str, Any]]) -> None:
    if not results:
        typer.echo("No QA results found.")
        return
    for result in results:
        output = result.get("canonical_result") or result.get("result") or {}
        verdict = output.get("verdict") or result.get("status", "unknown")
        summary = output.get("summary") or result.get("error")
        subject_id = result.get("subject_id", "-")
        agent = result.get("agent_name") or result.get("qa_agent_id") or "-"
        stale = " stale" if result.get("stale") is True else ""
        line = f"{subject_id}\t{agent}\t{verdict}{stale}"
        typer.echo(f"{line}\t{summary}" if summary else line)


def _wait_for_results(
    platform: PlatformClient,
    result_ids: list[str],
    timeout: float,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while True:
        results = cast(
            "list[dict[str, Any]]",
            platform.get(
                "/qa-agents/results/resources",
                params={"result_ids": result_ids},
            ),
        )
        if len(results) == len(result_ids) and all(
            result["status"] in _TERMINAL_STATUSES for result in results
        ):
            return results
        if time.monotonic() >= deadline:
            raise HudTimeoutError(f"Timed out after {timeout:g}s waiting for QA runs.")
        time.sleep(_POLL_INTERVAL_SECONDS)


@qa_app.command("agents")
def list_agents(
    subject_type: ResourceSubjectType = typer.Option(  # noqa: B008
        ResourceSubjectType.environment,
        "--subject-type",
        help="Resource scope: environment, taskset, or task.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output the machine-readable response."),
    limit: int = typer.Option(50, "--limit", min=1, max=500, help="Maximum agents to return."),
    offset: int = typer.Option(0, "--offset", min=0, help="Number of agents to skip."),
) -> None:
    """List QA agents available for a resource type."""
    require_api_key("use platform QA agents")
    response = cast(
        "dict[str, Any]",
        PlatformClient.from_settings().get(
            "/qa-agents",
            params={
                "subject_type": subject_type.value,
                "limit": limit,
                "offset": offset,
            },
        ),
    )
    if json_output:
        typer.echo(json.dumps(response, indent=2, sort_keys=True, default=str))
        return
    agents = response["items"]
    if not agents:
        typer.echo(f"No {subject_type.value} QA agents found.")
        return
    for agent in agents:
        typer.echo(
            f"{agent.get('id', '-')}\t{agent.get('name', '-')}\t"
            f"{agent.get('subject_type', '-')}\t{agent.get('model_name') or '-'}"
        )


@qa_app.command("run")
def run_agent(
    agent_id: str = typer.Argument(..., help="QA agent UUID."),
    subject_ids: list[str] = typer.Argument(  # noqa: B008
        ...,
        help="One or more Environment, Taskset, or Task UUIDs.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Create a fresh attempt even when current evidence already exists.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for every launched analysis to finish.",
    ),
    timeout: float = typer.Option(
        900,
        "--timeout",
        min=1,
        help="Maximum seconds to wait for QA execution.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable results."),
) -> None:
    """Run one QA agent against resource subjects."""
    require_api_key("use platform QA agents")
    platform = PlatformClient.from_settings()
    runs = cast(
        "list[dict[str, Any]]",
        platform.post(
            f"/qa-agents/{agent_id}/run-resources",
            json={"subject_ids": subject_ids, "overwrite": overwrite},
        ),
    )
    if not wait:
        if json_output:
            typer.echo(json.dumps(runs, indent=2, sort_keys=True, default=str))
        else:
            _render_results(runs)
        return

    results = _wait_for_results(platform, [str(run["id"]) for run in runs], timeout)
    if json_output:
        typer.echo(json.dumps(results, indent=2, sort_keys=True, default=str))
    else:
        _render_results(results)
    if any(
        result["status"] == "error"
        or (result.get("canonical_result") or {}).get("verdict") != "passed"
        for result in results
    ):
        raise typer.Exit(1)


@qa_app.command("results")
def list_results(
    subject_type: ResourceSubjectType = typer.Argument(  # noqa: B008
        ...,
        help="Resource scope: environment, taskset, or task.",
    ),
    subject_ids: list[str] = typer.Argument(  # noqa: B008
        ...,
        help="One or more Environment, Taskset, or Task UUIDs.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable results."),
) -> None:
    """Inspect QA results attached to resource subjects."""
    require_api_key("use platform QA agents")
    results = cast(
        "list[dict[str, Any]]",
        PlatformClient.from_settings().get(
            "/qa-agents/results/resources",
            params={
                "subject_type": subject_type.value,
                "subject_ids": subject_ids,
            },
        ),
    )
    if json_output:
        typer.echo(json.dumps(results, indent=2, sort_keys=True, default=str))
    else:
        _render_results(results)
