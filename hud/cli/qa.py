"""List, run, and inspect trace-level platform QA agents."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import typer

from hud.cli.utils.api import require_api_key
from hud.utils.exceptions import HudTimeoutError
from hud.utils.platform import PlatformClient

_POLL_INTERVAL_SECONDS = 2.0
_TERMINAL_STATUSES = {"completed", "error"}
_TRACE_SUBJECT = "trace"

qa_app = typer.Typer(
    name="qa",
    help="List, run, and inspect trace-level platform QA agents.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


def _platform() -> PlatformClient:
    require_api_key("use platform QA agents")
    return PlatformClient.from_settings()


def _print_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _print_agent(agent: dict[str, Any]) -> None:
    typer.echo(f"{agent.get('name', '-')}\t{agent.get('id', '-')}")


def _print_results(results: list[dict[str, Any]], *, json_output: bool) -> None:
    if json_output:
        _print_json(results)
        return
    if not results:
        typer.echo("No QA results found.")
        return
    for result in results:
        output = result.get("canonical_result") or result.get("result") or {}
        verdict = output.get("verdict") or result.get("status", "unknown")
        summary = output.get("summary") or result.get("error")
        subject_id = result.get("subject_trace_id") or "-"
        agent = result.get("agent_name") or result.get("qa_agent_id") or "-"
        stale = " stale" if result.get("stale") is True else ""
        line = f"{subject_id}\t{agent}\t{verdict}{stale}"
        typer.echo(f"{line}\t{summary}" if summary else line)


def _require_trace_agent(platform: PlatformClient, agent_id: str) -> None:
    agent = cast("dict[str, Any]", platform.get(f"/qa-agents/{agent_id}"))
    if agent.get("subject_type") != _TRACE_SUBJECT:
        typer.echo(
            f"Agent {agent_id} is a {agent.get('subject_type')} QA agent. "
            "The CLI currently supports trace agents only.",
            err=True,
        )
        raise typer.Exit(1)


def _latest_agent_result(
    results: list[dict[str, Any]],
    *,
    agent_id: str,
    trace_id: str,
) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for result in results:
        if (
            result.get("qa_agent_id") == agent_id
            and str(result.get("subject_trace_id")) == trace_id
        ):
            latest = result
    return latest


def _wait_for_results(
    platform: PlatformClient,
    timeout: float,
    *,
    trace_ids: list[str],
    agent_id: str,
    launched: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    launched_id_by_trace = {str(run["subject_trace_id"]): str(run["id"]) for run in launched}
    deadline = time.monotonic() + timeout
    while True:
        listed = cast(
            "list[dict[str, Any]]",
            platform.get("/qa-agents/results", params={"subject_trace_ids": trace_ids}),
        )
        by_id = {str(result["id"]): result for result in listed}
        selected: list[dict[str, Any]] = []
        ready = True
        for trace_id in trace_ids:
            result_id = launched_id_by_trace.get(trace_id)
            result = (
                by_id.get(result_id)
                if result_id is not None
                else _latest_agent_result(listed, agent_id=agent_id, trace_id=trace_id)
            )
            if result is None or result["status"] not in _TERMINAL_STATUSES:
                ready = False
                break
            selected.append(result)
        if ready:
            return selected
        if time.monotonic() >= deadline:
            raise HudTimeoutError(f"Timed out after {timeout:g}s waiting for QA runs.")
        time.sleep(_POLL_INTERVAL_SECONDS)


@qa_app.callback(invoke_without_command=True)
def list_agents(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Output the machine-readable response."),
    limit: int = typer.Option(50, "--limit", min=1, max=500, help="Maximum agents to return."),
    offset: int = typer.Option(0, "--offset", min=0, help="Number of agents to skip."),
) -> None:
    """List trace QA agents available to this team."""
    if ctx.invoked_subcommand is not None:
        return
    response = cast(
        "dict[str, Any]",
        _platform().get(
            "/qa-agents",
            params={"subject_type": _TRACE_SUBJECT, "limit": limit, "offset": offset},
        ),
    )
    if json_output:
        _print_json(response)
        return
    agents = response["items"]
    if not agents:
        typer.echo("No trace QA agents found.")
        return
    for agent in agents:
        _print_agent(agent)


@qa_app.command("run")
def run_agent(
    agent_id: str = typer.Argument(..., help="QA agent UUID."),
    trace_ids: list[str] = typer.Argument(  # noqa: B008
        ...,
        help="One or more trace UUIDs.",
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
    """Run one trace QA agent against the given traces."""
    platform = _platform()
    _require_trace_agent(platform, agent_id)
    runs = cast(
        "list[dict[str, Any]]",
        platform.post(
            f"/qa-agents/{agent_id}/run",
            json={"trace_ids": trace_ids, "overwrite": overwrite},
        ),
    )
    if not wait:
        _print_results(runs, json_output=json_output)
        return

    results = _wait_for_results(
        platform,
        timeout,
        trace_ids=trace_ids,
        agent_id=agent_id,
        launched=runs,
    )
    _print_results(results, json_output=json_output)
    if any(
        result["status"] == "error"
        or (result.get("canonical_result") or {}).get("verdict") != "passed"
        for result in results
    ):
        raise typer.Exit(1)


@qa_app.command("results")
def list_results(
    trace_ids: list[str] = typer.Argument(  # noqa: B008
        ...,
        help="One or more trace UUIDs.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable results."),
) -> None:
    """Inspect QA results attached to the given traces."""
    results = cast(
        "list[dict[str, Any]]",
        _platform().get("/qa-agents/results", params={"subject_trace_ids": trace_ids}),
    )
    _print_results(results, json_output=json_output)
