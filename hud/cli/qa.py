"""List, run, and inspect HUD's QA checks on evaluation traces and tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any
from uuid import UUID

import typer
from pydantic import ConfigDict, with_config
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing_extensions import TypedDict

from hud.cli import (
    CLI,
    CliError,
    Result,
    map_exception,
)
from hud.settings import settings
from hud.utils.exceptions import HudException, HudTimeoutError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

hud_console = HUDConsole()

_POLL_INTERVAL_SECONDS = 2.0
_TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})
_VERDICT_STYLES = {"passed": "green", "failed": "red", "unknown": "yellow", "error": "magenta"}
_SEVERITY_STYLES = {"error": "red", "warning": "yellow"}


class Level(StrEnum):
    """What a QA check judges: one evaluation trace, or one task's current version."""

    trace = "trace"
    task = "task"


@with_config(ConfigDict(extra="allow"))
class _Check(TypedDict):
    key: str
    subject_type: str
    title: str
    question: str


@with_config(ConfigDict(extra="allow"))
class _Catalog(TypedDict):
    checks: list[_Check]


class _Results(TypedDict):
    results: list[dict[str, Any]]


_IDS_FIELD = {Level.trace: "trace_ids", Level.task: "task_ids"}


@dataclass(frozen=True)
class _Finding:
    summary: str
    finding_type: str
    severity: str
    action: str | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Presentation:
    """How one QA result reads: ``kind`` is ``pending``, ``result`` or ``unknown``; ``tag``
    the verdict."""

    kind: str
    tag: str
    summary: str | None = None
    confidence: str | None = None
    findings: tuple[_Finding, ...] = ()


def _confidence(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("confidence")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, int | float):
        return f"{round((raw / 100 if raw > 1 else raw) * 100)}%"
    return None


def _from_result(result: dict[str, Any]) -> _Presentation:
    """Read a ``qa_agent_result.v1`` result, the one shape the QA API returns."""
    summary = result.get("summary")
    return _Presentation(
        "result",
        str(result["verdict"]),
        summary=summary.strip() or None if isinstance(summary, str) else None,
        confidence=_confidence(result.get("metadata") or {}),
        findings=tuple(
            _Finding(
                summary=finding["summary"],
                finding_type=finding["finding_type"],
                severity=finding["severity"],
                action=finding.get("recommended_action") or None,
                evidence=tuple(finding.get("evidence_refs") or ()),
            )
            for finding in result.get("findings") or ()
        ),
    )


def _presentation(row: dict[str, Any]) -> _Presentation:
    """How a ``/v2/qa/results`` row reads; the row's own verdict wins over its output's."""
    status = str(row["status"])
    if status == "error":
        error = row.get("error")
        return _Presentation(
            "unknown", "error", summary=str(error) if error else "QA did not judge this subject."
        )
    if status != "completed":
        return _Presentation("pending", "unknown")
    result = row.get("result")
    view = _from_result(result) if result is not None else _Presentation("unknown", "unknown")
    verdict = row.get("verdict")
    return replace(view, tag=str(verdict)) if verdict else view


def _verdict(result: dict[str, Any], view: _Presentation) -> Text:
    if view.kind == "pending":
        return Text(str(result["status"]), style="dim")
    return Text(view.tag, style=_VERDICT_STYLES.get(view.tag, "yellow"))


def _headline(text: str | None) -> Markdown | str:
    """The first non-empty line of a markdown summary."""
    line = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
    return Markdown(line) if line else "-"


def _level(result: dict[str, Any]) -> Level:
    return Level(result["subject_type"])


def _subject_id(result: dict[str, Any]) -> str:
    """The trace or task a result judges."""
    if _level(result) is Level.task:
        return str(result["subject_task_id"])
    return str(result["subject_trace_id"])


def _print_results(results: list[dict[str, Any]], level: Level) -> None:
    """One table row per result: trace or task, check, verdict, and the summary's first line."""
    if not results:
        hud_console.stdout.print("[yellow]No QA results found.[/yellow]")
        return
    hud_console.stdout.print(Panel.fit("[bold cyan]QA Results[/bold cyan]", border_style="cyan"))
    table = Table()
    table.add_column(level.value.capitalize(), style="blue", no_wrap=True)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Verdict", no_wrap=True)
    table.add_column("Summary")
    for result in results:
        view = _presentation(result)
        table.add_row(
            _subject_id(result),
            str(result["check_key"]),
            _verdict(result, view),
            _headline(view.summary or result.get("note")),
        )
    hud_console.stdout.print(table)
    results_command = (
        "hud qa results --level task <task-id>"
        if level is Level.task
        else "hud qa results <trace-id>"
    )
    hud_console.stdout.print(f"\n[dim]Tip: {results_command} for summaries and findings[/dim]")


def _finding_panel(index: int, finding: _Finding) -> Panel:
    body: list[RenderableType] = []
    if finding.action:
        body.append(Text.assemble(("Action: ", "bold"), finding.action))
    details = [f"type: {finding.finding_type}", f"severity: {finding.severity}"]
    if finding.evidence:
        details.append(f"evidence: {', '.join(finding.evidence)}")
    body.append(Text("\n".join(details), style="dim"))
    return Panel(
        Group(*body),
        title=Text(f"{index}. {finding.summary}", style="bold"),
        title_align="left",
        border_style=_SEVERITY_STYLES.get(finding.severity, "dim"),
        padding=(0, 1),
    )


def _print_result(result: dict[str, Any], web_url: str) -> None:
    """One result in full: verdict, details, markdown summary, findings, and links."""
    level = _level(result)
    subject_id = _subject_id(result)
    view = _presentation(result)
    hud_console.stdout.print(
        Panel.fit(
            Text.assemble((str(result["check_key"]), "bold cyan"), " ", (subject_id, "dim")),
            border_style="cyan",
        )
    )
    label = "Status" if view.kind == "pending" else "Verdict"
    hud_console.stdout.print(Text.assemble((f"{label}: ", "bold"), _verdict(result, view)))

    details: list[tuple[str, str]] = []
    if level is Level.task:
        details.append(("Task version", str(result["subject_task_version_id"])))
    if result["source"] != "analysis":
        details.append(("Source", str(result["source"])))
    if view.confidence:
        details.append(("Confidence", view.confidence))
    if result.get("note"):
        details.append(("Note", str(result["note"])))
    for name, value in details:
        hud_console.stdout.print(Text(f"{name}: {value}", style="dim"))

    if view.summary:
        hud_console.stdout.print(
            Panel(
                Markdown(view.summary),
                title=Text("Summary", style="bold"),
                title_align="left",
                border_style="dim",
                padding=(0, 1),
            )
        )
    for index, finding in enumerate(view.findings, start=1):
        hud_console.stdout.print(_finding_panel(index, finding))

    links: list[str] = []
    if level is Level.trace:
        links.append(f"View: {web_url}/trace/{subject_id}")
    if result.get("analysis_trace_id"):
        links.append(f"Analysis: {web_url}/trace/{result['analysis_trace_id']}")
    if links:
        hud_console.stdout.print()
    for link in links:
        hud_console.stdout.print(f"[dim]{link}[/dim]")


def _results(
    platform: PlatformClient,
    level: Level,
    ids: list[str],
    *,
    history: bool = False,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {_IDS_FIELD[level]: ids}
    if history:
        params["include_history"] = True
    return platform.get("/qa/results", _Results, params=params)["results"]


def _checks_level(platform: PlatformClient, check_keys: list[str]) -> Level:
    """The one level every named check runs at."""
    catalog = platform.get("/qa/checks", _Catalog)["checks"]
    levels = {check["key"]: check["subject_type"] for check in catalog}
    unknown = [key for key in check_keys if key not in levels]
    if unknown:
        raise CliError(
            error="usage",
            message=f"Unknown QA check: {', '.join(unknown)}. `hud qa list` shows them.",
            input={"check_keys": check_keys},
        )
    found = {levels[key] for key in check_keys}
    if len(found) > 1:
        raise CliError(
            error="usage",
            message="Run trace-level and task-level checks separately.",
            input={"check_keys": check_keys},
        )
    level = found.pop()
    if level not in {member.value for member in Level}:
        raise CliError(
            error="failure",
            message=f"This hud version cannot run {level}-level QA checks.",
            input={"check_keys": check_keys},
            suggestion="Upgrade hud.",
        )
    return Level(level)


qa_app = CLI(
    name="qa",
    help="List, run, and inspect HUD's QA checks on evaluation traces and tasks.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=False,
)


@qa_app.command("list")
def list_command(
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one check key per line, with no headers (for piping)."
    ),
    level: Level | None = typer.Option(
        None, "--level", case_sensitive=False, help="Only the checks that run at this level."
    ),
) -> Any:
    """List the QA checks HUD runs on evaluation traces and tasks.

    [not dim]Examples:
        hud qa list
        hud qa list --level task
        hud qa list --json
        hud qa list --quiet[/not dim]
    """
    catalog = PlatformClient.from_settings().get("/qa/checks", _Catalog)
    checks = [
        check for check in catalog["checks"] if level is None or check["subject_type"] == level
    ]
    response = {**catalog, "checks": checks}
    if quiet:
        for check in checks:
            typer.echo(check["key"])
        return response
    if not checks:
        hud_console.stdout.print("[yellow]No QA checks are available.[/yellow]")
        return response

    hud_console.stdout.print(Panel.fit("[bold cyan]QA Checks[/bold cyan]", border_style="cyan"))
    table = Table()
    table.add_column("Key", style="blue", no_wrap=True)
    table.add_column("Level", no_wrap=True)
    table.add_column("Title", style="cyan")
    table.add_column("Question")
    for check in checks:
        table.add_row(check["key"], check["subject_type"], check["title"], check["question"])
    hud_console.stdout.print(table)
    hud_console.stdout.print(
        "\n[dim]Tip: hud qa run <check>[,<check>...] <trace-or-task-id>... to run checks[/dim]"
    )
    return response


@qa_app.callback(invoke_without_command=True)
def qa_command(
    ctx: typer.Context,
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one check key per line, with no headers (for piping)."
    ),
    level: Level | None = typer.Option(
        None, "--level", case_sensitive=False, help="Only the checks that run at this level."
    ),
) -> Any:
    """List QA checks (same as hud qa list), or run and inspect them with a subcommand.

    [not dim]Examples:
        hud qa
        hud qa list --json
        hud qa run failure_analysis <trace-id>
        hud qa run task_reward_exploit <task-id>[/not dim]
    """
    if ctx.invoked_subcommand is not None:
        return None
    return list_command(quiet=quiet, level=level)


@qa_app.command("run")
def run_checks(
    checks: str = typer.Argument(
        ...,
        help="QA check key from `hud qa list`, or several separated by commas.",
    ),
    ids: list[UUID] = typer.Argument(
        ...,
        help=(
            "Completed evaluation trace UUIDs for trace-level checks, or task UUIDs for "
            "task-level checks (at most 100)."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Run again even where a check's current result is completed or in flight.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for every check to finish on every trace or task.",
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
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts (required in non-interactive terminals).",
    ),
) -> Any:
    """Run QA checks on completed evaluation traces or on tasks as one run.

    The checks decide what the IDs are: trace-level checks take traces, task-level
    checks take tasks and judge each task's current version.

    [not dim]Examples:
        hud qa run failure_analysis <trace-id>
        hud qa run reward_hacking,false_positive <trace-id> <trace-id> --no-wait --json
        hud qa run task_reward_exploit,task_failure_analysis <task-id>
        hud qa run failure_analysis <trace-id> --yes --json
        hud qa run failure_analysis <trace-id> --dry-run --json[/not dim]
    """
    check_keys = list(dict.fromkeys(key.strip() for key in checks.split(",") if key.strip()))
    if not check_keys:
        raise CliError(
            error="usage",
            message="Name at least one QA check; `hud qa list` shows them.",
            input={"checks": checks},
        )
    subject_ids = [str(subject_id) for subject_id in ids]
    platform = PlatformClient.from_settings()
    level = _checks_level(platform, check_keys)
    ids_field = _IDS_FIELD[level]
    if dry_run:
        typer.echo(
            f"--dry-run: would run {len(check_keys)} check(s) on {len(subject_ids)} {level}(s)"
        )
        return {
            "dry_run": True,
            "action": "qa_run",
            "check_keys": check_keys,
            "level": level.value,
            ids_field: subject_ids,
            "overwrite": overwrite,
            "wait": wait,
        }
    CLI.confirm_or_abort(
        f"Run {len(check_keys)} check(s) on {len(subject_ids)} {level}(s)?", yes=yes, default=True
    )

    try:
        results = platform.post(
            "/qa/runs",
            _Results,
            json={"check_keys": check_keys, ids_field: subject_ids, "overwrite": overwrite},
        )["results"]
    except HudException as exc:
        raise map_exception(exc, input={"check_keys": check_keys, ids_field: subject_ids}) from exc
    if not wait:
        _print_results(results, level)
        return results

    requested = {(subject_id, check_key) for subject_id in subject_ids for check_key in check_keys}
    deadline = time.monotonic() + timeout
    while len(results) < len(requested) or any(
        result["status"] not in _TERMINAL_STATUSES for result in results
    ):
        if time.monotonic() >= deadline:
            raise HudTimeoutError(f"Timed out after {timeout:g}s waiting for QA checks.")
        time.sleep(_POLL_INTERVAL_SECONDS)
        results = [
            result
            for result in _results(platform, level, subject_ids)
            if (_subject_id(result), result["check_key"]) in requested
        ]

    _print_results(results, level)
    failed = any(_presentation(result).tag != "passed" for result in results)
    return Result(results) if failed else results


@qa_app.command("results")
def list_results(
    ids: list[UUID] = typer.Argument(
        ...,
        help="Evaluation trace UUIDs, or task UUIDs with --level task (at most 100).",
    ),
    level: Level = typer.Option(
        Level.trace, "--level", case_sensitive=False, help="Whether the IDs are traces or tasks."
    ),
    history: bool = typer.Option(
        False,
        "--history",
        help=(
            "Every result, newest first, instead of the one in effect per check. "
            "For tasks this includes results on earlier versions."
        ),
    ),
) -> Any:
    """Inspect the QA result in effect for every check on the given traces or tasks.

    A task's result in effect is the latest one on its current version.

    [not dim]Examples:
        hud qa results <trace-id>
        hud qa results --level task <task-id>
        hud qa results --level task <task-id> --history --json[/not dim]
    """
    results = _results(
        PlatformClient.from_settings(),
        level,
        [str(subject_id) for subject_id in ids],
        history=history,
    )
    if not results:
        hud_console.stdout.print("[yellow]No QA results found.[/yellow]")
        return results

    web_url = settings.hud_web_url.rstrip("/")
    for index, result in enumerate(results):
        if index:
            hud_console.stdout.print()
        _print_result(result, web_url)
    return results
