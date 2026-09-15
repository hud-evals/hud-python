"""List, run, and inspect trace-level platform QA agents."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, cast

import typer
from rich.panel import Panel
from rich.text import Text

from hud.cli.app import (
    CLI,
    CliError,
    Result,
    map_exception,
)
from hud.settings import settings
from hud.utils.exceptions import HudException, HudTimeoutError
from hud.utils.hud_console import DIM, GOLD, GREEN, RED, SECONDARY, HUDConsole
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()

_POLL_INTERVAL_SECONDS = 2.0
_TERMINAL_STATUSES = {"completed", "error"}
_TRACE_SUBJECT = "trace"
_RESULT_LINE_CAP = 12
_QA_V1 = frozenset({"passed", "failed", "unknown"})
_BOOLEAN_KEYS = (
    ("is_false_negative", "False Negative"),
    ("is_false_positive", "False Positive"),
    ("is_reward_hacking", "Reward Hacking"),
    ("is_prompt_misaligned", "Prompt Misaligned"),
)
_CAUSE = {
    "agent": "Agent failure",
    "eval": "Evaluation failure",
    "platform": "Platform failure",
}


@dataclass(frozen=True)
class QaFinding:
    title: str
    description: str
    fault: str | None = None


@dataclass(frozen=True)
class QaPresentation:
    kind: str
    tag: str
    label: str
    answer: str | None
    summary: str | None
    confidence: str | None
    findings: tuple[QaFinding, ...]


def is_standard_result_blob(text: str) -> bool:
    parsed = _loads(text)
    return parsed is not None and _from_blob(parsed).kind != "unknown"


def presentation_for_result(row: dict[str, Any]) -> QaPresentation:
    status = str(row.get("status") or "")
    error = row.get("error")
    if status == "error":
        return _view("unknown", "failed", summary=str(error) if error else "QA run failed.")
    if status and status != "completed":
        return _view("pending", "unknown", label=status, summary=str(error) if error else None)
    payload = row.get("canonical_result")
    if not isinstance(payload, dict):
        payload = row.get("result")
    blob = _unwrap(payload) if isinstance(payload, dict | str) else None
    if blob is None:
        return _view("unknown", "unknown", summary=str(error) if error else None)
    return _from_blob(blob)


def _from_blob(parsed: dict[str, Any]) -> QaPresentation:
    summary = _text(parsed.get("summary"), parsed.get("reasoning"))
    confidence = _confidence(parsed.get("confidence"))

    verdict = parsed.get("verdict")
    findings_raw = parsed.get("findings")
    if (
        isinstance(verdict, str)
        and verdict in _QA_V1
        and (parsed.get("schema_version") == "qa_agent_result.v1" or isinstance(findings_raw, list))
    ):
        findings = _findings(findings_raw if isinstance(findings_raw, list) else [], "summary")
        return _view(
            "qa_result",
            verdict,
            summary=summary,
            confidence=confidence,
            findings=findings,
        )

    for key, label in _BOOLEAN_KEYS:
        if key not in parsed:
            continue
        value = parsed[key]
        if not isinstance(value, bool):
            return _view("unknown", "unknown", label=label, summary=summary, confidence=confidence)
        return _view(
            "boolean",
            "failed" if value else "passed",
            label=label,
            answer="yes" if value else "no",
            summary=summary,
            confidence=confidence,
        )

    if isinstance(parsed.get("problems"), list):
        findings = _findings(parsed["problems"], "problem", "title")
        owners = {_finding_owner(item.fault) for item in findings}
        if not findings:
            cause, tag = "No failure", "passed"
        elif len(owners) != 1:
            cause, tag = "Mixed failure", "failed"
        elif "unclear" in owners:
            cause, tag = "Unclear", "failed"
        else:
            cause, tag = _CAUSE[next(iter(owners))], "failed"
        return _view(
            "problems",
            tag,
            label="Failure Analysis",
            answer=cause,
            summary=summary,
            confidence=confidence,
            findings=findings,
        )

    return _view("unknown", "unknown", summary=summary, confidence=confidence)


def _view(
    kind: str,
    tag: str,
    *,
    label: str = "QA Result",
    answer: str | None = None,
    summary: str | None = None,
    confidence: str | None = None,
    findings: tuple[QaFinding, ...] = (),
) -> QaPresentation:
    return QaPresentation(kind, tag, label, answer, summary, confidence, findings)


def _unwrap(payload: dict[str, Any] | str) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None = _loads(payload) if isinstance(payload, str) else payload
    if parsed is None:
        return None
    for key in ("output", "content"):
        raw = parsed.get(key)
        if isinstance(raw, str):
            inner = _loads(raw)
            if inner is not None:
                parsed = inner
    return parsed if isinstance(parsed, dict) else None


def _loads(raw: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _text(*values: Any) -> str | None:
    parts: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    return "\n\n".join(parts) if parts else None


def _confidence(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        lower = raw.strip().lower()
        labels = {"high", "medium", "low", "very high", "very low"}
        return lower if lower in labels else raw.strip()
    if isinstance(raw, int | float):
        value = raw / 100 if raw > 1 else raw
        return f"{round(value * 100)}%"
    return None


def _findings(items: list[Any], *title_keys: str) -> tuple[QaFinding, ...]:
    findings: list[QaFinding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = next(
            (
                item[key].strip()
                for key in title_keys
                if isinstance(item.get(key), str) and item[key].strip()
            ),
            "",
        )
        if not title:
            continue
        description = item.get("description")
        fault = item.get("fault") if isinstance(item.get("fault"), str) else None
        findings.append(
            QaFinding(
                title=title,
                description=description.strip() if isinstance(description, str) else "",
                fault=fault,
            )
        )
    return tuple(findings)


def _finding_owner(fault: str | None) -> str:
    if fault is None:
        return "unclear"
    owner = fault.strip().lower()
    if owner in _CAUSE:
        return owner
    return "unclear"


qa_app = CLI(
    name="qa",
    help="List, run, and inspect trace-level platform QA agents.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


def _platform() -> PlatformClient:
    return PlatformClient.from_settings()


def _print_agent(agent: dict[str, Any]) -> None:
    typer.echo(f"{agent.get('name', '-')}\t{agent.get('id', '-')}")


def _print_results_human(results: list[dict[str, Any]]) -> None:
    if not results:
        typer.echo("No QA results found.")
        return
    for result in results:
        view = presentation_for_result(result)
        verdict = result.get("status", "unknown") if view.kind == "pending" else view.tag
        summary = view.summary or result.get("error")
        subject_id = result.get("subject_trace_id") or "-"
        agent = result.get("agent_name") or result.get("qa_agent_id") or "-"
        stale = " stale" if result.get("stale") is True else ""
        line = f"{subject_id}\t{agent}\t{verdict}{stale}"
        typer.echo(f"{line}\t{summary}" if summary else line)


def _print_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _print_results_human(results)
    return results


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
        if next_seq <= since_seq:
            raise ValueError("QA rollout pagination did not advance")
        since_seq = next_seq


def _tool_command(event: dict[str, Any]) -> str:
    args = event.get("arguments") or {}
    if not isinstance(args, dict):
        return str(args)
    commands = args.get("commands")
    if isinstance(commands, list):
        return "\n".join(str(item) for item in commands)
    claims = args.get("claims")
    if claims:
        return str(claims)
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


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
            if isinstance(text, str) and is_standard_result_blob(text):
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
            hud_console.stdout.print(
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
            hud_console.stdout.print(
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
            hud_console.stdout.print(
                Panel(
                    Text(_tool_command(event) if args else name, style=DIM),
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
    view = presentation_for_result(result)
    hud_console.header(agent, icon="", stderr=False)
    if view.kind == "pending":
        hud_console.status_item(
            "status",
            str(result.get("status") or "unknown"),
            status="info",
            stderr=False,
        )
    else:
        if view.tag == "passed":
            status = "success"
        elif view.tag == "failed":
            status = "error"
        else:
            status = "info"
        hud_console.status_item("verdict", view.tag, status=status, stderr=False)
        if view.kind == "boolean" and view.answer:
            hud_console.dim_info(view.label.lower(), view.answer, stderr=False)
        elif view.answer:
            hud_console.dim_info("cause", view.answer, stderr=False)
    hud_console.dim_info("trace", subject_id, stderr=False)
    if view.confidence:
        hud_console.dim_info("confidence", view.confidence, stderr=False)
    if result.get("stale") is True:
        hud_console.warning(
            "This result is stale relative to the current agent config.",
            stderr=False,
        )
    if view.summary:
        hud_console.stdout.print(
            Panel(
                Text(view.summary),
                title=_bold_title("Summary"),
                border_style=GOLD,
                padding=(0, 1),
            )
        )
    for index, finding in enumerate(view.findings, start=1):
        body = Text(finding.description)
        if finding.fault:
            if finding.description:
                body.append("\n\n")
            body.append(f"fault: {finding.fault}", style=DIM)
        hud_console.stdout.print(
            Panel(
                body,
                title=_bold_title(f"{index}. {finding.title}"),
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
        raise CliError(
            error="usage",
            message=(
                f"Agent {agent_id} is a {agent.get('subject_type')} QA agent. "
                "The CLI currently supports trace agents only."
            ),
            input={"agent_id": agent_id, "subject_type": agent.get("subject_type")},
            suggestion="Use a trace QA agent id from `hud qa list --json`.",
        )


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


def _list_agents(*, quiet: bool, limit: int, offset: int) -> dict[str, Any]:
    response = cast(
        "dict[str, Any]",
        _platform().get(
            "/qa-agents",
            params={"subject_type": _TRACE_SUBJECT, "limit": limit, "offset": offset},
        ),
    )

    def _render(payload: dict[str, Any]) -> None:
        agents = payload["items"]
        if not agents:
            typer.echo("No trace QA agents found.")
            return
        for agent in agents:
            _print_agent(agent)

    if quiet:
        for agent in response["items"]:
            if agent.get("id"):
                typer.echo(agent["id"])
    else:
        _render(response)
    return response


@qa_app.command("list")
def list_command(
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
    limit: int = typer.Option(50, "--limit", min=1, max=500, help="Maximum agents to return."),
    offset: int = typer.Option(0, "--offset", min=0, help="Number of agents to skip."),
) -> Any:
    """List trace QA agents available to this team.

    [not dim]Examples:
        hud qa list
        hud qa list --json
        hud qa list --quiet[/not dim]
    """
    return _list_agents(quiet=quiet, limit=limit, offset=offset)


@qa_app.callback(invoke_without_command=True)
def qa_command(
    ctx: typer.Context,
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
    limit: int = typer.Option(50, "--limit", min=1, max=500, help="Maximum agents to return."),
    offset: int = typer.Option(0, "--offset", min=0, help="Number of agents to skip."),
) -> Any:
    """List trace QA agents, or run and inspect them.

    Without a verb, lists available agents. ``hud qa`` is an alias for ``hud qa list``.

    [not dim]Examples:
        hud qa
        hud qa list --json
        hud qa run <agent-id> <trace-id>[/not dim]
    """
    if ctx.invoked_subcommand is not None:
        return None
    return _list_agents(quiet=quiet, limit=limit, offset=offset)


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
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
) -> Any:
    """Run one trace QA agent against the given traces.

    [not dim]Examples:
        hud qa run <agent-id> <trace-id>
        hud qa run <agent-id> <trace-id> --json --no-wait
        hud qa run <agent-id> <trace-id> --dry-run --json[/not dim]
    """
    if dry_run:
        plan = {
            "dry_run": True,
            "action": "qa_run",
            "agent_id": agent_id,
            "trace_ids": trace_ids,
            "overwrite": overwrite,
            "wait": wait,
        }
        typer.echo(
            f"--dry-run: would run agent {plan['agent_id']} on {len(plan['trace_ids'])} trace(s)"
        )
        return plan
    platform = _platform()
    _require_trace_agent(platform, agent_id)
    try:
        runs = cast(
            "list[dict[str, Any]]",
            platform.post(
                f"/qa-agents/{agent_id}/run",
                json={"trace_ids": trace_ids, "overwrite": overwrite},
            ),
        )
    except HudException as exc:
        raise map_exception(exc, input={"agent_id": agent_id, "trace_ids": trace_ids}) from exc
    if not wait:
        return _print_results(runs)

    results = _wait_for_results(
        platform,
        timeout,
        trace_ids=trace_ids,
        agent_id=agent_id,
        launched=runs,
    )
    _print_results_human(results)
    if any(
        result["status"] == "error" or presentation_for_result(result).tag != "passed"
        for result in results
    ):
        return Result(results)
    return results


@qa_app.command("results")
def list_results(
    trace_ids: list[str] = typer.Argument(  # noqa: B008
        ...,
        help="One or more trace UUIDs.",
    ),
    rollout: bool = typer.Option(
        False,
        "--rollout",
        help="Show the sanitized analysis trajectory (agent turns and tool calls).",
    ),
) -> Any:
    """Inspect QA results for the given traces. Pass --rollout for the trajectory."""
    platform = _platform()
    results = cast(
        "list[dict[str, Any]]",
        platform.get("/qa-agents/results", params={"subject_trace_ids": trace_ids}),
    )

    if not results:
        typer.echo("No QA results found.")
        return results
    for result in results:
        events: list[dict[str, Any]] | None = None
        result_id = result.get("id")
        if rollout and result_id:
            events = _fetch_rollout(platform, str(result_id))
        _print_result_tui(result, events)
    return results
