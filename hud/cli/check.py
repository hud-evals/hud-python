"""Run the platform's environment readiness checklist."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import typer

from hud.cli.utils.api import require_api_key
from hud.cli.utils.registry import resolve_registry_environments
from hud.utils.exceptions import HudRequestError
from hud.utils.platform import PlatformClient

_POLL_INTERVAL_SECONDS = 2.0


def _request_error(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(2)


def _resolve_environment_id(platform: PlatformClient, reference: str) -> str:
    matches = resolve_registry_environments(platform, reference)
    if not matches:
        _request_error(f"No environment matched {reference!r}.")
    if len(matches) > 1:
        names = ", ".join(f"{match.name} ({match.short_id})" for match in matches)
        _request_error(f"{reference!r} matches multiple environments: {names}")
    return matches[0].id


def _probe_is_running(report: dict[str, Any]) -> bool:
    criteria = report.get("criteria")
    if not isinstance(criteria, list):
        return False
    for raw_criterion in cast("list[object]", criteria):
        if not isinstance(raw_criterion, dict):
            continue
        criterion = cast("dict[str, Any]", raw_criterion)
        if criterion.get("check_key") != "environment_probe_task_runs":
            continue
        evidence = criterion.get("evidence")
        return (
            criterion.get("status") == "unknown"
            and isinstance(evidence, dict)
            and isinstance(cast("dict[str, Any]", evidence).get("execution_trace_id"), str)
        )
    return False


def check_command(
    environment: str = typer.Argument(..., help="Environment name or UUID."),
    json_output: bool = typer.Option(False, "--json", help="Output the machine-readable report."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Create a fresh check attempt instead of reusing an identical result.",
    ),
    active: bool = typer.Option(
        False,
        "--active",
        help="Start the declared worker-backed readiness probe.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Probe model override; otherwise use the platform default.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Poll an active probe until its trace reaches a terminal state.",
    ),
    timeout: float = typer.Option(
        600,
        "--timeout",
        min=1,
        help="Maximum seconds to wait for an active probe.",
    ),
) -> None:
    """Check whether a deployed HUD environment is ready for task execution."""
    try:
        require_api_key("check an environment")
    except typer.Exit as exc:
        raise typer.Exit(2) from exc

    platform = PlatformClient.from_settings()
    if model is not None and not active:
        _request_error("--model requires --active")
    try:
        environment_id = _resolve_environment_id(platform, environment)
        request: dict[str, object] = {
            "environment_id": environment_id,
            "overwrite": overwrite,
        }
        if active:
            request["run_active_probe"] = True
        if model is not None:
            request["probe_model"] = model
        raw_report = platform.post(
            "/checks/environment-readiness",
            json=request,
        )
    except HudRequestError as exc:
        _request_error(str(exc))

    if not isinstance(raw_report, dict):
        typer.echo("Platform returned an invalid readiness report.", err=True)
        raise typer.Exit(3)
    report = cast("dict[str, Any]", raw_report)
    deadline = time.monotonic() + timeout
    while active and wait and _probe_is_running(report) and time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SECONDS)
        poll_request = {**request, "overwrite": False}
        try:
            raw_report = platform.post("/checks/environment-readiness", json=poll_request)
        except HudRequestError as exc:
            _request_error(str(exc))
        if not isinstance(raw_report, dict):
            typer.echo("Platform returned an invalid readiness report.", err=True)
            raise typer.Exit(3)
        report = cast("dict[str, Any]", raw_report)
    poll_timed_out = active and wait and _probe_is_running(report)
    status = report.get("status")
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        human_report = report.get("human_report")
        typer.echo(
            human_report
            if isinstance(human_report, str)
            else str(report.get("summary") or "No readiness summary returned."),
        )

    if poll_timed_out:
        typer.echo(f"Timed out after {timeout:g}s waiting for the readiness probe.", err=True)
        raise typer.Exit(3)
    if status == "passed":
        return
    if status == "error":
        raise typer.Exit(3)
    if status in {"failed", "unknown"}:
        raise typer.Exit(1)
    typer.echo(f"Platform returned unknown readiness status: {status!r}", err=True)
    raise typer.Exit(3)
