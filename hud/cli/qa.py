"""List, run, and inspect HUD's QA checks on evaluation traces."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Any, cast
from uuid import UUID  # noqa: TC003 - typer resolves argument annotations at runtime

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
from hud.utils.hud_console import DIM, GOLD, HUDConsole
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()

_POLL_INTERVAL_SECONDS = 2.0
_TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})
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


def _legacy_output(raw: dict[str, Any]) -> dict[str, Any]:
    """A pre-backfill QA agent output, unwrapped from its ``output`` and ``content`` strings."""
    parsed = raw
    for key in ("output", "content"):
        inner = parsed.get(key)
        unwrapped = _loads(inner) if isinstance(inner, str) else None
        if unwrapped is not None:
            parsed = unwrapped
    return parsed


def presentation_for_result(row: dict[str, Any]) -> QaPresentation:
    """How a ``/v2/qa/results`` row reads; the row's own verdict wins over its output's."""
    status = str(row["status"])
    if status == "error":
        error = row.get("error")
        return QaPresentation(
            "unknown", "failed", summary=str(error) if error else "QA run failed."
        )
    if status != "completed":
        return QaPresentation("pending", "unknown", label=status)
    result = row.get("result")
    legacy = row.get("legacy_result")
    if isinstance(result, dict):
        view = _from_blob(cast("dict[str, Any]", result))
    elif isinstance(legacy, dict):
        view = _from_blob(_legacy_output(cast("dict[str, Any]", legacy)))
    else:
        view = QaPresentation("unknown", "unknown")
    verdict = row.get("verdict")
    return replace(view, tag=str(verdict)) if verdict else view


def _print_results(results: list[dict[str, Any]]) -> None:
    """One tab-separated line per result: trace, check, verdict, summary on one line."""
    if not results:
        typer.echo("No QA results found.")
        return
    for result in results:
        view = presentation_for_result(result)
        verdict = result["status"] if view.kind == "pending" else view.tag
        summary = " ".join(str(view.summary or result.get("note") or "").split())
        line = f"{result['subject_trace_id']}\t{result['check_key']}\t{verdict}"
        typer.echo(f"{line}\t{summary}" if summary else line)


def _results(platform: PlatformClient, trace_ids: list[str]) -> list[dict[str, Any]]:
    response = cast("dict[str, Any]", platform.get("/qa/results", params={"trace_ids": trace_ids}))
    return cast("list[dict[str, Any]]", response["results"])


qa_app = CLI(
    name="qa",
    help="List, run, and inspect HUD's QA checks on evaluation traces.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


@qa_app.command("list")
def list_command(
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one check key per line, with no headers (for piping)."
    ),
) -> Any:
    """List the QA checks HUD runs on evaluation traces.

    [not dim]Examples:
        hud qa list
        hud qa list --json
        hud qa list --quiet[/not dim]
    """
    response = cast("dict[str, Any]", PlatformClient.from_settings().get("/qa/checks"))
    checks = response["checks"]
    if quiet:
        for check in checks:
            typer.echo(check["key"])
    elif not checks:
        typer.echo("No QA checks are available.")
    else:
        for check in checks:
            typer.echo(f"{check['title']}\t{check['key']}\t{check['question']}")
    return response


@qa_app.callback(invoke_without_command=True)
def qa_command(
    ctx: typer.Context,
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one check key per line, with no headers (for piping)."
    ),
) -> Any:
    """List QA checks, or run and inspect them.

    Without a verb, lists the checks. ``hud qa`` is an alias for ``hud qa list``.

    [not dim]Examples:
        hud qa
        hud qa list --json
        hud qa run failure_analysis <trace-id>[/not dim]
    """
    if ctx.invoked_subcommand is not None:
        return None
    return list_command(quiet=quiet)


@qa_app.command("run")
def run_checks(
    checks: str = typer.Argument(
        ...,
        help="QA check key from `hud qa list`, or several separated by commas.",
    ),
    trace_ids: list[UUID] = typer.Argument(  # noqa: B008
        ...,
        help="One or more completed evaluation trace UUIDs (at most 100).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Run again even where a check's current result is completed or in flight.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for every check to finish on every trace.",
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
    """Run QA checks on completed evaluation traces as one run.

    [not dim]Examples:
        hud qa run failure_analysis <trace-id>
        hud qa run reward_hacking,false_positive <trace-id> <trace-id> --no-wait --json
        hud qa run failure_analysis <trace-id> --dry-run --json[/not dim]
    """
    check_keys = list(dict.fromkeys(key.strip() for key in checks.split(",") if key.strip()))
    if not check_keys:
        raise CliError(
            error="usage",
            message="Name at least one QA check; `hud qa list` shows them.",
            input={"checks": checks},
        )
    traces = [str(trace_id) for trace_id in trace_ids]
    if dry_run:
        typer.echo(f"--dry-run: would run {len(check_keys)} check(s) on {len(traces)} trace(s)")
        return {
            "dry_run": True,
            "action": "qa_run",
            "check_keys": check_keys,
            "trace_ids": traces,
            "overwrite": overwrite,
            "wait": wait,
        }

    platform = PlatformClient.from_settings()
    try:
        response = cast(
            "dict[str, Any]",
            platform.post(
                "/qa/runs",
                json={"check_keys": check_keys, "trace_ids": traces, "overwrite": overwrite},
            ),
        )
    except HudException as exc:
        raise map_exception(exc, input={"check_keys": check_keys, "trace_ids": traces}) from exc
    results = cast("list[dict[str, Any]]", response["results"])
    if not wait:
        _print_results(results)
        return results

    requested = {(trace_id, check_key) for trace_id in traces for check_key in check_keys}
    deadline = time.monotonic() + timeout
    while len(results) < len(requested) or any(
        result["status"] not in _TERMINAL_STATUSES for result in results
    ):
        if time.monotonic() >= deadline:
            raise HudTimeoutError(f"Timed out after {timeout:g}s waiting for QA checks.")
        time.sleep(_POLL_INTERVAL_SECONDS)
        results = [
            result
            for result in _results(platform, traces)
            if (result["subject_trace_id"], result["check_key"]) in requested
        ]

    _print_results(results)
    failed = any(presentation_for_result(result).tag != "passed" for result in results)
    return Result(results) if failed else results


@qa_app.command("results")
def list_results(
    trace_ids: list[UUID] = typer.Argument(  # noqa: B008
        ...,
        help="One or more evaluation trace UUIDs (at most 100).",
    ),
) -> Any:
    """Inspect the QA result in effect for every check on the given traces."""
    results = _results(PlatformClient.from_settings(), [str(trace_id) for trace_id in trace_ids])
    if not results:
        typer.echo("No QA results found.")
        return results

    web_url = settings.hud_web_url.rstrip("/")
    for result in results:
        subject_id = str(result["subject_trace_id"])
        view = presentation_for_result(result)
        hud_console.header(str(result["check_key"]), icon="", stderr=False)
        if view.kind == "pending":
            hud_console.status_item("status", str(result["status"]), status="info", stderr=False)
        else:
            status = {"passed": "success", "failed": "error"}.get(view.tag, "info")
            hud_console.status_item("verdict", view.tag, status=status, stderr=False)
            if view.kind == "boolean" and view.answer:
                hud_console.dim_info(view.label.lower(), view.answer, stderr=False)
            elif view.answer:
                hud_console.dim_info("cause", view.answer, stderr=False)
        hud_console.dim_info("trace", subject_id, stderr=False)
        if result["source"] != "analysis":
            hud_console.dim_info("source", str(result["source"]), stderr=False)
        if view.confidence:
            hud_console.dim_info("confidence", view.confidence, stderr=False)
        if result.get("note"):
            hud_console.dim_info("note", str(result["note"]), stderr=False)
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
        if result.get("analysis_trace_id"):
            hud_console.dim_info(
                "analysis", f"{web_url}/trace/{result['analysis_trace_id']}", stderr=False
            )
        hud_console.link(f"{web_url}/trace/{subject_id}", stderr=False)
    return results
