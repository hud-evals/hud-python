"""List, run, and inspect trace-level platform QA agents."""

from __future__ import annotations

import json
import time
from typing import Any, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from hud.cli.utils.api import require_api_key
from hud.settings import settings
from hud.utils.exceptions import HudTimeoutError
from hud.utils.hud_console import DIM, GOLD, GREEN, RED, SECONDARY, HUDConsole
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()
_console = Console()

_POLL_INTERVAL_SECONDS = 2.0
_TERMINAL_STATUSES = {"completed", "error"}
_TRACE_SUBJECT = "trace"
_RESULT_LINE_CAP = 12

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


def _as_problems(payload: dict[str, Any]) -> list[Any]:
    problems = payload.get("problems")
    if isinstance(problems, list):
        return problems
    findings = payload.get("findings")
    return findings if isinstance(findings, list) else []


def _parse_analysis_blob(
    payload: dict[str, Any],
    *,
    fallback_verdict: str,
) -> dict[str, Any] | None:
    if (
        payload.get("verdict")
        or payload.get("summary")
        or payload.get("problems")
        or payload.get("findings")
    ):
        return {
            "verdict": str(payload.get("verdict") or fallback_verdict),
            "summary": payload.get("summary"),
            "problems": _as_problems(payload),
            "confidence": payload.get("confidence"),
        }
    content = payload.get("content")
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {
        "verdict": str(parsed.get("verdict") or fallback_verdict),
        "summary": parsed.get("summary"),
        "problems": _as_problems(parsed),
        "confidence": parsed.get("confidence"),
    }


def _analysis_view(result: dict[str, Any]) -> dict[str, Any]:
    """Verdict, summary, and findings from a canonical row or a JSON content blob."""
    fallback = str(result.get("status") or "unknown")
    for payload in (result.get("canonical_result"), result.get("result")):
        if isinstance(payload, dict):
            parsed = _parse_analysis_blob(payload, fallback_verdict=fallback)
            if parsed is not None:
                return parsed
    error = result.get("error")
    return {
        "verdict": fallback,
        "summary": str(error) if error else None,
        "problems": [],
        "confidence": None,
    }


def _fetch_rollout(platform: PlatformClient, result_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    since_seq = -1
    while True:
        page = cast(
            "dict[str, Any]",
            platform.get(
                f"/qa-agents/results/{result_id}/rollout",
                params={"since_seq": since_seq, "limit": 100},
            ),
        )
        events.extend(cast("list[dict[str, Any]]", page.get("events") or []))
        if not page.get("has_more"):
            return events
        next_seq = int(page["next_seq"])
        if next_seq == since_seq:
            return events
        since_seq = next_seq


def _fmt_args(args: Any) -> str:
    if isinstance(args, dict):
        return ", ".join(f"{k}={v!r}" for k, v in args.items())
    return str(args)


def _is_analysis_json(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and "summary" in parsed


def _tool_command(event: dict[str, Any]) -> str:
    args = event.get("arguments") or {}
    if isinstance(args, dict):
        commands = args.get("commands")
        if isinstance(commands, list):
            return "\n".join(str(item) for item in commands)
        claims = args.get("claims")
        if claims:
            return str(claims)
    return _fmt_args(args)


def _capped_text(value: str, *, max_lines: int) -> Text:
    lines = value.splitlines() or [value]
    shown = lines[:max_lines]
    text = Text("\n".join(shown))
    extra = len(lines) - len(shown)
    if extra > 0:
        text.append(f"\n… {extra} more lines", style=DIM)
    return text


def _bold_title(label: str) -> Text:
    return Text(label, style="bold")


def _render_qa_rollout(events: list[dict[str, Any]]) -> None:
    turn = 0
    for event in events:
        kind = event.get("kind")
        if kind == "agent_message":
            text = event.get("text")
            reasoning = event.get("reasoning")
            if isinstance(text, str) and _is_analysis_json(text):
                continue
            if not text and not reasoning:
                continue
            turn += 1
            body = Text()
            if reasoning:
                body.append(str(reasoning), style=f"italic {DIM}")
                if text:
                    body.append("\n")
            if text:
                body.append(str(text))
            _console.print(
                Panel(
                    body,
                    title=_bold_title(f"Turn {turn} · agent"),
                    border_style=SECONDARY,
                    padding=(0, 1),
                )
            )
        elif kind in ("tool_call", "tool_result"):
            name = str(event.get("tool_name") or event.get("name") or "tool")
            error = event.get("error")
            result = event.get("result_text") or event.get("result") or ""
            body = Text(_tool_command(event))
            if error:
                body.append(f"\n\nerror: {error}", style=RED)
                border = RED
            else:
                if result:
                    body.append("\n\n")
                    body.append_text(_capped_text(str(result), max_lines=_RESULT_LINE_CAP))
                border = GREEN
            _console.print(
                Panel(
                    body,
                    title=_bold_title(name),
                    border_style=border,
                    padding=(0, 1),
                )
            )
        elif kind == "subagent":
            name = str(event.get("agent_name") or "subagent")
            args = event.get("arguments") or {}
            _console.print(
                Panel(
                    Text(_fmt_args(args) if args else name, style=DIM),
                    title=_bold_title(name),
                    border_style=GOLD,
                    padding=(0, 1),
                )
            )


def _print_result_tui(
    result: dict[str, Any],
    events: list[dict[str, Any]] | None,
) -> None:
    agent = str(result.get("agent_name") or result.get("qa_agent_id") or "QA")
    subject_id = str(result.get("subject_trace_id") or "-")
    analysis = _analysis_view(result)
    verdict = str(analysis["verdict"])
    if verdict == "passed":
        status = "success"
    elif verdict in {"failed", "error"}:
        status = "error"
    else:
        status = "info"
    hud_console.header(agent, icon="", stderr=False)
    hud_console.status_item("verdict", verdict, status=status, stderr=False)
    hud_console.dim_info("trace", subject_id, stderr=False)
    if analysis.get("confidence"):
        hud_console.dim_info("confidence", str(analysis["confidence"]), stderr=False)
    if result.get("stale") is True:
        hud_console.warning(
            "This result is stale relative to the current agent config.",
            stderr=False,
        )
    summary = analysis.get("summary")
    if summary:
        _console.print(
            Panel(
                Text(str(summary)),
                title=_bold_title("Summary"),
                border_style=GOLD,
                padding=(0, 1),
            )
        )
    problems = analysis.get("problems") or []
    for index, problem in enumerate(problems, start=1):
        if not isinstance(problem, dict):
            continue
        title = str(problem.get("problem") or problem.get("title") or f"Finding {index}")
        description = str(problem.get("description") or "")
        fault = problem.get("fault")
        body = Text(description)
        if fault:
            body.append(f"\n\nfault: {fault}", style=DIM)
        _console.print(
            Panel(
                body,
                title=_bold_title(f"{index}. {title}"),
                border_style=GOLD,
                padding=(0, 1),
            )
        )
    if events is None:
        hud_console.dim_info("trajectory", "hidden; pass --rollout to show", stderr=False)
    elif events:
        hud_console.section_title("Rollout", stderr=False)
        _render_qa_rollout(events)
    web = settings.hud_web_url.rstrip("/")
    hud_console.link(f"{web}/trace/{subject_id}", stderr=False)


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
    json_output: bool = typer.Option(False, "--json", help="Output the machine-readable response."),
    rollout: bool = typer.Option(
        False,
        "--rollout",
        help="Show the sanitized analysis trajectory (agent turns and tool calls).",
    ),
) -> None:
    """Inspect QA results for the given traces. Pass --rollout for the trajectory."""
    platform = _platform()
    results = cast(
        "list[dict[str, Any]]",
        platform.get("/qa-agents/results", params={"subject_trace_ids": trace_ids}),
    )
    if json_output:
        _print_results(results, json_output=True)
        return
    if not results:
        typer.echo("No QA results found.")
        return
    for result in results:
        events: list[dict[str, Any]] | None = None
        result_id = result.get("id")
        if rollout and result_id:
            events = _fetch_rollout(platform, str(result_id))
        _print_result_tui(result, events)
