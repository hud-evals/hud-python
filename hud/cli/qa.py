"""Author, run, and inspect trace-level platform QA agents."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import typer

from hud.cli.models import _resolve_model_id
from hud.cli.utils.api import require_api_key
from hud.utils.exceptions import HudTimeoutError
from hud.utils.platform import PlatformClient

_POLL_INTERVAL_SECONDS = 2.0
_TERMINAL_STATUSES = {"completed", "error"}
_TRACE_SUBJECT = "trace"

qa_app = typer.Typer(
    name="qa",
    help="Author, run, and inspect trace-level platform QA agents.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _platform() -> PlatformClient:
    require_api_key("use platform QA agents")
    return PlatformClient.from_settings()


def _print_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _print_agent(agent: dict[str, Any], *, json_output: bool = False) -> None:
    if json_output:
        _print_json(agent)
        return
    typer.echo(
        f"{agent.get('id', '-')}\t{agent.get('name', '-')}\t"
        f"{agent.get('subject_type', '-')}\t{agent.get('model_name') or '-'}"
    )


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


def _parse_args(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        typer.echo("--args must be a JSON object.", err=True)
        raise typer.Exit(1)
    return parsed


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


@qa_app.command("templates")
def list_templates(
    json_output: bool = typer.Option(False, "--json", help="Output the machine-readable response."),
    limit: int = typer.Option(500, "--limit", min=1, max=1000, help="Maximum templates to return."),
) -> None:
    """List templates that can be used to author a trace QA agent."""
    templates = cast(
        "list[dict[str, Any]]",
        _platform().get(
            "/qa-agents/scenarios",
            params={"subject_type": _TRACE_SUBJECT, "limit": limit},
        ),
    )
    if json_output:
        _print_json(templates)
        return
    if not templates:
        typer.echo("No trace QA templates found.")
        return
    for template in templates:
        typer.echo(
            f"{template.get('id', '-')}\t{template.get('name', '-')}\t"
            f"{template.get('registry_display_name') or '-'}"
        )


@qa_app.command("agents")
def list_agents(
    json_output: bool = typer.Option(False, "--json", help="Output the machine-readable response."),
    limit: int = typer.Option(50, "--limit", min=1, max=500, help="Maximum agents to return."),
    offset: int = typer.Option(0, "--offset", min=0, help="Number of agents to skip."),
) -> None:
    """List trace QA agents available to this team."""
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


@qa_app.command("create")
def create_agent(
    name: str = typer.Option(..., "--name", help="Display name for the agent."),
    template: str = typer.Option(
        ...,
        "--template",
        help="Template UUID from `hud qa templates`.",
    ),
    model: str = typer.Option(
        ...,
        "--model",
        help="Model UUID or slug from `hud models list`.",
    ),
    description: str | None = typer.Option(
        None,
        "--description",
        help="Optional agent description.",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        min=1,
        help="Maximum analysis steps.",
    ),
    args: str | None = typer.Option(None, "--args", help="JSON object of template partial args."),
    public: bool = typer.Option(False, "--public", help="Make the agent visible to other teams."),
    json_output: bool = typer.Option(False, "--json", help="Output the created agent as JSON."),
) -> None:
    """Create a trace QA agent."""
    body: dict[str, Any] = {
        "name": name,
        "scenario_id": template,
        "model_id": _resolve_model_id(model),
        "subject_type": _TRACE_SUBJECT,
    }
    if description is not None:
        body["description"] = description
    if max_steps is not None:
        body["max_steps"] = max_steps
    if args is not None:
        body["partial_args"] = _parse_args(args)
    if public:
        body["public"] = True
    _print_agent(
        cast("dict[str, Any]", _platform().post("/qa-agents", json=body)),
        json_output=json_output,
    )


@qa_app.command("update")
def update_agent(
    agent_id: str = typer.Argument(..., help="QA agent UUID."),
    name: str | None = typer.Option(None, "--name", help="New display name."),
    model: str | None = typer.Option(None, "--model", help="Model UUID or slug."),
    description: str | None = typer.Option(None, "--description", help="New description."),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        min=1,
        help="Maximum analysis steps.",
    ),
    args: str | None = typer.Option(None, "--args", help="JSON object of template partial args."),
    public: bool | None = typer.Option(
        None,
        "--public/--private",
        help="Whether this agent is visible to other teams.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output the updated agent as JSON."),
) -> None:
    """Update a trace QA agent's configuration."""
    platform = _platform()
    _require_trace_agent(platform, agent_id)
    body: dict[str, Any] = {}
    if name is not None:
        body["name"] = name
    if model is not None:
        body["model_id"] = _resolve_model_id(model)
    if description is not None:
        body["description"] = description
    if max_steps is not None:
        body["max_steps"] = max_steps
    if args is not None:
        body["partial_args"] = _parse_args(args)
    if public is not None:
        body["public"] = public
    if not body:
        typer.echo("Pass at least one field to update.", err=True)
        raise typer.Exit(1)
    _print_agent(
        cast("dict[str, Any]", platform.patch(f"/qa-agents/{agent_id}", json=body)),
        json_output=json_output,
    )


@qa_app.command("delete")
def delete_agent(
    agent_id: str = typer.Argument(..., help="QA agent UUID."),
) -> None:
    """Delete a trace QA agent."""
    platform = _platform()
    _require_trace_agent(platform, agent_id)
    platform.delete(f"/qa-agents/{agent_id}")
    typer.echo(f"Deleted {agent_id}.")


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
