"""List, run, and inspect trace-level platform QA agents."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, cast

import typer
from rich.panel import Panel
from rich.text import Text

from hud.cli import (
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
_RESULT_LINE_CAP = 12
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
    """How one QA result reads: ``kind`` is the schema it matched, ``tag`` the verdict."""

    kind: str
    tag: str
    label: str = "QA Result"
    answer: str | None = None
    summary: str | None = None
    confidence: str | None = None
    findings: tuple[QaFinding, ...] = ()


def _loads(raw: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


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
        fault = item.get("fault")
        findings.append(
            QaFinding(
                title=title,
                description=description.strip() if isinstance(description, str) else "",
                fault=fault if isinstance(fault, str) else None,
            )
        )
    return tuple(findings)


def _from_blob(parsed: dict[str, Any]) -> QaPresentation:
    """Read a QA result blob in any of the shapes agents emit."""
    parts = [
        value.strip()
        for value in (parsed.get("summary"), parsed.get("reasoning"))
        if isinstance(value, str) and value.strip()
    ]
    summary = "\n\n".join(dict.fromkeys(parts)) or None

    confidence: str | None = None
    raw_confidence = parsed.get("confidence")
    if isinstance(raw_confidence, str) and raw_confidence.strip():
        lowered = raw_confidence.strip().lower()
        confidence = (
            lowered
            if lowered in {"high", "medium", "low", "very high", "very low"}
            else raw_confidence.strip()
        )
    elif isinstance(raw_confidence, int | float):
        share = raw_confidence / 100 if raw_confidence > 1 else raw_confidence
        confidence = f"{round(share * 100)}%"

    # qa_agent_result.v1: an explicit verdict plus findings.
    verdict = parsed.get("verdict")
    findings_raw = parsed.get("findings")
    if verdict in ("passed", "failed", "unknown") and (
        parsed.get("schema_version") == "qa_agent_result.v1" or isinstance(findings_raw, list)
    ):
        return QaPresentation(
            "qa_result",
            verdict,
            summary=summary,
            confidence=confidence,
            findings=_findings(findings_raw if isinstance(findings_raw, list) else [], "summary"),
        )

    # Boolean agents: one is_* flag answers a yes/no question.
    for key, label in _BOOLEAN_KEYS:
        if key not in parsed:
            continue
        value = parsed[key]
        if not isinstance(value, bool):
            return QaPresentation(
                "unknown", "unknown", label=label, summary=summary, confidence=confidence
            )
        return QaPresentation(
            "boolean",
            "failed" if value else "passed",
            label=label,
            answer="yes" if value else "no",
            summary=summary,
            confidence=confidence,
        )

    # Failure analysis: a list of problems, each attributed to a fault owner.
    if isinstance(parsed.get("problems"), list):
        findings = _findings(parsed["problems"], "problem", "title")
        owners = {
            (item.fault or "").strip().lower()
            if (item.fault or "").strip().lower() in _CAUSE
            else "unclear"
            for item in findings
        }
        if not findings:
            cause, tag = "No failure", "passed"
        elif len(owners) != 1:
            cause, tag = "Mixed failure", "failed"
        elif "unclear" in owners:
            cause, tag = "Unclear", "failed"
        else:
            cause, tag = _CAUSE[next(iter(owners))], "failed"
        return QaPresentation(
            "problems",
            tag,
            label="Failure Analysis",
            answer=cause,
            summary=summary,
            confidence=confidence,
            findings=findings,
        )

    return QaPresentation("unknown", "unknown", summary=summary, confidence=confidence)


def is_standard_result_blob(text: str) -> bool:
    parsed = _loads(text)
    return parsed is not None and _from_blob(parsed).kind != "unknown"


def presentation_for_result(row: dict[str, Any]) -> QaPresentation:
    status = str(row.get("status") or "")
    error = row.get("error")
    if status == "error":
        return QaPresentation(
            "unknown", "failed", summary=str(error) if error else "QA run failed."
        )
    if status and status != "completed":
        return QaPresentation(
            "pending", "unknown", label=status, summary=str(error) if error else None
        )
    payload = row.get("canonical_result")
    if not isinstance(payload, dict):
        payload = row.get("result")
    # The blob may arrive as a dict, a JSON string, or wrapped in an output/content string.
    parsed = _loads(payload) if isinstance(payload, str) else payload
    if isinstance(parsed, dict):
        for key in ("output", "content"):
            raw = parsed.get(key)
            inner = _loads(raw) if isinstance(raw, str) else None
            if inner is not None:
                parsed = inner
    if not isinstance(parsed, dict):
        return QaPresentation("unknown", "unknown", summary=str(error) if error else None)
    return _from_blob(parsed)


def _tool_command(event: dict[str, Any]) -> str:
    args = event.get("arguments") or {}
    if not isinstance(args, dict):
        return str(args)
    if isinstance(args.get("commands"), list):
        return "\n".join(str(item) for item in args["commands"])
    if args.get("claims"):
        return str(args["claims"])
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _print_results(results: list[dict[str, Any]]) -> None:
    """One tab-separated line per result: trace, agent, verdict, summary."""
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


qa_app = CLI(
    name="qa",
    help="List, run, and inspect trace-level platform QA agents.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


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
    response = cast(
        "dict[str, Any]",
        PlatformClient.from_settings().get(
            "/qa-agents", params={"subject_type": "trace", "limit": limit, "offset": offset}
        ),
    )
    agents = response["items"]
    if quiet:
        for agent in agents:
            if agent.get("id"):
                typer.echo(agent["id"])
    elif not agents:
        typer.echo("No trace QA agents found.")
    else:
        for agent in agents:
            typer.echo(f"{agent.get('name', '-')}\t{agent.get('id', '-')}")
    return response


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
    return list_command(quiet=quiet, limit=limit, offset=offset)


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
        typer.echo(f"--dry-run: would run agent {agent_id} on {len(trace_ids)} trace(s)")
        return {
            "dry_run": True,
            "action": "qa_run",
            "agent_id": agent_id,
            "trace_ids": trace_ids,
            "overwrite": overwrite,
            "wait": wait,
        }

    platform = PlatformClient.from_settings()
    agent = cast("dict[str, Any]", platform.get(f"/qa-agents/{agent_id}"))
    if agent.get("subject_type") != "trace":
        raise CliError(
            error="usage",
            message=(
                f"Agent {agent_id} is a {agent.get('subject_type')} QA agent. "
                "The CLI currently supports trace agents only."
            ),
            input={"agent_id": agent_id, "subject_type": agent.get("subject_type")},
            suggestion="Use a trace QA agent id from `hud qa list --json`.",
        )
    try:
        launched = cast(
            "list[dict[str, Any]]",
            platform.post(
                f"/qa-agents/{agent_id}/run",
                json={"trace_ids": trace_ids, "overwrite": overwrite},
            ),
        )
    except HudException as exc:
        raise map_exception(exc, input={"agent_id": agent_id, "trace_ids": trace_ids}) from exc
    if not wait:
        _print_results(launched)
        return launched

    # Poll until every trace has a terminal result: the launched run's own row when
    # the launch returned one, otherwise this agent's latest row for the trace.
    launched_by_trace = {str(run["subject_trace_id"]): str(run["id"]) for run in launched}
    deadline = time.monotonic() + timeout
    while True:
        listed = cast(
            "list[dict[str, Any]]",
            platform.get("/qa-agents/results", params={"subject_trace_ids": trace_ids}),
        )
        by_id = {str(result["id"]): result for result in listed}
        results: list[dict[str, Any]] = []
        for trace_id in trace_ids:
            if trace_id in launched_by_trace:
                result = by_id.get(launched_by_trace[trace_id])
            else:
                result = next(
                    (
                        row
                        for row in reversed(listed)
                        if row.get("qa_agent_id") == agent_id
                        and str(row.get("subject_trace_id")) == trace_id
                    ),
                    None,
                )
            if result is None or result["status"] not in ("completed", "error"):
                break
            results.append(result)
        if len(results) == len(trace_ids):
            break
        if time.monotonic() >= deadline:
            raise HudTimeoutError(f"Timed out after {timeout:g}s waiting for QA runs.")
        time.sleep(_POLL_INTERVAL_SECONDS)

    _print_results(results)
    failed = any(
        result["status"] == "error" or presentation_for_result(result).tag != "passed"
        for result in results
    )
    return Result(results) if failed else results


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
    platform = PlatformClient.from_settings()
    results = cast(
        "list[dict[str, Any]]",
        platform.get("/qa-agents/results", params={"subject_trace_ids": trace_ids}),
    )
    if not results:
        typer.echo("No QA results found.")
        return results

    for result in results:
        agent = str(result.get("agent_name") or result.get("qa_agent_id") or "QA")
        subject_id = str(result.get("subject_trace_id") or "-")
        view = presentation_for_result(result)
        hud_console.header(agent, icon="", stderr=False)
        if view.kind == "pending":
            hud_console.status_item(
                "status", str(result.get("status") or "unknown"), status="info", stderr=False
            )
        else:
            status = {"passed": "success", "failed": "error"}.get(view.tag, "info")
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
                "This result is stale relative to the current agent config.", stderr=False
            )
        if view.summary:
            hud_console.stdout.print(
                Panel(
                    Text(view.summary),
                    title=Text("Summary", style="bold"),
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
                    title=Text(f"{index}. {finding.title}", style="bold"),
                    border_style=GOLD,
                    padding=(0, 1),
                )
            )

        if not rollout or not result.get("id"):
            hud_console.dim_info("trajectory", "hidden; pass --rollout to show", stderr=False)
        else:
            events: list[dict[str, Any]] = []
            since_seq = -1
            while True:
                page = cast(
                    "dict[str, Any]",
                    platform.get(
                        f"/qa-agents/results/{result['id']}/rollout",
                        params={"since_seq": since_seq, "limit": 100},
                    ),
                )
                events.extend(page.get("events") or [])
                if not page.get("has_more"):
                    break
                if int(page["next_seq"]) <= since_seq:
                    raise ValueError("QA rollout pagination did not advance")
                since_seq = int(page["next_seq"])

            if events:
                hud_console.section_title("Rollout", stderr=False)
            turn = 0
            for event in events:
                kind = event.get("kind")
                if kind == "agent_message":
                    text = event.get("text")
                    reasoning = event.get("reasoning")
                    # The agent's final structured verdict is shown above, not as a turn.
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
                            title=Text(f"Turn {turn} · agent", style="bold"),
                            border_style=SECONDARY,
                            padding=(0, 1),
                        )
                    )
                elif kind in ("tool_call", "tool_result"):
                    name = str(event.get("tool_name") or event.get("name") or "tool")
                    body = Text(_tool_command(event))
                    if event.get("error"):
                        body.append(f"\n\nerror: {event['error']}", style=RED)
                        border = RED
                    else:
                        output = str(event.get("result_text") or event.get("result") or "")
                        if output:
                            lines = output.splitlines() or [output]
                            body.append("\n\n" + "\n".join(lines[:_RESULT_LINE_CAP]))
                            if len(lines) > _RESULT_LINE_CAP:
                                body.append(
                                    f"\n… {len(lines) - _RESULT_LINE_CAP} more lines", style=DIM
                                )
                        border = GREEN
                    hud_console.stdout.print(
                        Panel(
                            body,
                            title=Text(name, style="bold"),
                            border_style=border,
                            padding=(0, 1),
                        )
                    )
                elif kind == "subagent":
                    name = str(event.get("agent_name") or "subagent")
                    hud_console.stdout.print(
                        Panel(
                            Text(
                                _tool_command(event) if event.get("arguments") else name,
                                style=DIM,
                            ),
                            title=Text(name, style="bold"),
                            border_style=GOLD,
                            padding=(0, 1),
                        )
                    )
        hud_console.link(f"{settings.hud_web_url.rstrip('/')}/trace/{subject_id}", stderr=False)
    return results
