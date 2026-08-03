"""Discover, run, and inspect resource-scoped platform QA agents."""

from __future__ import annotations

import json
import time
from typing import Any, NoReturn, cast

import typer

from hud.cli.utils.api import require_api_key
from hud.utils.exceptions import HudNetworkError, HudRequestError, HudTimeoutError
from hud.utils.platform import PlatformClient

_POLL_INTERVAL_SECONDS = 2.0
_RESOURCE_SUBJECT_TYPES = {"environment", "taskset"}
_TERMINAL_STATUSES = {"completed", "error"}

qa_app = typer.Typer(
    name="qa",
    help="Discover, run, and inspect platform QA agents.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _request_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(2)


def _execution_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(3)


def _platform() -> PlatformClient:
    try:
        require_api_key("use platform QA agents")
    except typer.Exit as exc:
        raise typer.Exit(2) from exc
    return PlatformClient.from_settings()


def _subject_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _RESOURCE_SUBJECT_TYPES:
        choices = ", ".join(sorted(_RESOURCE_SUBJECT_TYPES))
        _request_error(f"Subject type must be one of: {choices}.")
    return normalized


def _dict_list(value: object, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        _execution_error(f"Platform returned invalid {label}.")
    return cast("list[dict[str, Any]]", value)


def _print_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _canonical_id(value: object) -> str:
    """Match UUID-like identifiers independently of accepted hex casing."""
    return str(value).casefold()


def _selected_trace_ids(
    runs: list[dict[str, Any]],
    *,
    agent_id: str,
    subject_ids: list[str],
) -> dict[str, str]:
    selected = {
        _canonical_id(run["subject_id"]): _canonical_id(run["analysis_trace_id"])
        for run in runs
        if isinstance(run.get("subject_id"), str)
        and isinstance(run.get("analysis_trace_id"), str)
        and _canonical_id(run.get("qa_agent_id")) == _canonical_id(agent_id)
    }
    expected_subject_ids = {_canonical_id(subject_id) for subject_id in subject_ids}
    if (
        len(runs) != len(subject_ids)
        or len(selected) != len(runs)
        or set(selected) != expected_subject_ids
    ):
        _execution_error("Platform did not select exactly one QA result per requested resource.")
    return selected


def _result_verdict(result: dict[str, Any]) -> tuple[str, str | None]:
    for result_field in ("canonical_result", "result"):
        canonical = result.get(result_field)
        if isinstance(canonical, dict):
            canonical_dict = cast("dict[str, Any]", canonical)
            verdict = canonical_dict.get("verdict")
            summary = canonical_dict.get("summary")
            if isinstance(verdict, str):
                return verdict, summary if isinstance(summary, str) else None
    status = result.get("status")
    error = result.get("error")
    return (
        status if isinstance(status, str) else "unknown",
        error if isinstance(error, str) else None,
    )


def _render_results(results: list[dict[str, Any]]) -> None:
    if not results:
        typer.echo("No QA results found.")
        return
    for result in results:
        verdict, summary = _result_verdict(result)
        subject_id = result.get("subject_id", "-")
        agent = result.get("agent_name") or result.get("qa_agent_id") or "-"
        stale = " stale" if result.get("stale") is True else ""
        line = f"{subject_id}\t{agent}\t{verdict}{stale}"
        typer.echo(f"{line}\t{summary}" if summary else line)


def _matching_subject_results(
    raw_results: object,
    *,
    agent_id: str,
    subject_ids: list[str],
    selected_trace_ids: dict[str, str],
) -> list[dict[str, Any]]:
    results = _dict_list(raw_results, label="QA results")
    expected_subject_ids = {_canonical_id(subject_id) for subject_id in subject_ids}
    matched: dict[str, dict[str, Any]] = {}
    for result in results:
        subject_id = _canonical_id(result.get("subject_id"))
        if (
            _canonical_id(result.get("qa_agent_id")) != _canonical_id(agent_id)
            or subject_id not in expected_subject_ids
        ):
            continue
        selected_trace_id = selected_trace_ids.get(subject_id)
        if (
            selected_trace_id is not None
            and _canonical_id(result.get("analysis_trace_id")) != selected_trace_id
        ):
            continue
        matched[subject_id] = result
    return [
        matched[canonical_id]
        for subject_id in subject_ids
        if (canonical_id := _canonical_id(subject_id)) in matched
    ]


def _all_terminal(results: list[dict[str, Any]], subject_ids: list[str]) -> bool:
    statuses = {_canonical_id(result.get("subject_id")): result.get("status") for result in results}
    return all(
        statuses.get(_canonical_id(subject_id)) in _TERMINAL_STATUSES for subject_id in subject_ids
    )


def _result_exit_code(results: list[dict[str, Any]]) -> int:
    if any(result.get("status") == "error" for result in results):
        return 3
    verdicts = [_result_verdict(result)[0] for result in results]
    if any(verdict in {"failed", "unknown"} for verdict in verdicts):
        return 1
    if any(verdict != "passed" for verdict in verdicts):
        return 3
    return 0


@qa_app.command("agents")
def list_agents(
    subject_type: str = typer.Option(
        "environment",
        "--subject-type",
        help="Resource scope: environment or taskset.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output the machine-readable response."),
    limit: int = typer.Option(50, "--limit", min=1, max=500, help="Maximum agents to return."),
    offset: int = typer.Option(0, "--offset", min=0, help="Number of agents to skip."),
) -> None:
    """List QA agents available for a resource type."""
    platform = _platform()
    normalized_type = _subject_type(subject_type)
    try:
        response = platform.get(
            "/qa-agents",
            params={"subject_type": normalized_type, "limit": limit, "offset": offset},
        )
    except (HudNetworkError, HudTimeoutError) as exc:
        _execution_error(str(exc))
    except HudRequestError as exc:
        _request_error(str(exc))
    if not isinstance(response, dict) or not isinstance(response.get("items"), list):
        typer.echo("Platform returned an invalid QA agent list.", err=True)
        raise typer.Exit(3)
    if json_output:
        _print_json(response)
        return
    agents = _dict_list(response["items"], label="QA agent list")
    if not agents:
        typer.echo(f"No {normalized_type} QA agents found.")
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
        help="One or more Environment or Taskset UUIDs.",
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
    """Run one QA agent against Environment or Taskset subjects."""
    platform = _platform()
    try:
        raw_agent = platform.get(f"/qa-agents/{agent_id}")
        if (
            not isinstance(raw_agent, dict)
            or raw_agent.get("subject_type") not in _RESOURCE_SUBJECT_TYPES
        ):
            _execution_error("Platform returned an invalid resource QA agent.")
        agent_subject_type = str(raw_agent["subject_type"])
        raw_runs = platform.post(
            f"/qa-agents/{agent_id}/run-resources",
            json={"subject_ids": subject_ids, "overwrite": overwrite},
        )
    except (HudNetworkError, HudTimeoutError) as exc:
        _execution_error(str(exc))
    except HudRequestError as exc:
        _request_error(str(exc))
    runs = _dict_list(raw_runs, label="QA launch response")
    selected_trace_ids = _selected_trace_ids(
        runs,
        agent_id=agent_id,
        subject_ids=subject_ids,
    )
    if not wait:
        if json_output:
            _print_json(runs)
        else:
            _render_results(runs)
        return

    deadline = time.monotonic() + timeout
    results: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        try:
            raw_results = platform.get(
                "/qa-agents/results/resources",
                params={
                    "subject_type": agent_subject_type,
                    "subject_ids": subject_ids,
                },
            )
        except (HudNetworkError, HudTimeoutError) as exc:
            _execution_error(str(exc))
        except HudRequestError as exc:
            _request_error(str(exc))
        results = _matching_subject_results(
            raw_results,
            agent_id=agent_id,
            subject_ids=subject_ids,
            selected_trace_ids=selected_trace_ids,
        )
        if _all_terminal(results, subject_ids):
            break
        time.sleep(_POLL_INTERVAL_SECONDS)
    else:
        if json_output:
            _print_json(results)
        else:
            _render_results(results)
        _execution_error(f"Timed out after {timeout:g}s waiting for QA runs.")

    if json_output:
        _print_json(results)
    else:
        _render_results(results)
    exit_code = _result_exit_code(results)
    if exit_code:
        raise typer.Exit(exit_code)


@qa_app.command("results")
def list_results(
    subject_type: str = typer.Argument(..., help="Resource scope: environment or taskset."),
    subject_ids: list[str] = typer.Argument(  # noqa: B008
        ...,
        help="One or more Environment or Taskset UUIDs.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable results."),
) -> None:
    """Inspect QA results attached to Environment or Taskset subjects."""
    platform = _platform()
    normalized_type = _subject_type(subject_type)
    try:
        raw_results = platform.get(
            "/qa-agents/results/resources",
            params={"subject_type": normalized_type, "subject_ids": subject_ids},
        )
    except (HudNetworkError, HudTimeoutError) as exc:
        _execution_error(str(exc))
    except HudRequestError as exc:
        _request_error(str(exc))
    results = _dict_list(raw_results, label="QA results")
    if json_output:
        _print_json(results)
    else:
        _render_results(results)
