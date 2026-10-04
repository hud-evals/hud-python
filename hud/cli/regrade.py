"""``hud regrade`` — regrade completed runs without re-running the agent."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import typer
from pydantic import AnyUrl, TypeAdapter

from hud.agents.types import AgentStep, ToolStep
from hud.cli import CLI, CliError
from hud.cli.eval import Placement, TcpUrl, load_taskset, named_placement
from hud.eval import HostedRuntime, Run, Runtime
from hud.settings import settings
from hud.telemetry.span import normalize_trace_id
from hud.types import Step, Trace
from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from hud.eval import Provider

hud_console = HUDConsole()

regrade_app = CLI(
    name="regrade",
    help="Regrade completed runs with the current task template.",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Source → Step subclass for span reconstruction.
# SubagentStep carries a nested Trace requiring recursive reconstruction; it
# falls back to base Step (losing subagent-specific fields but keeping skeleton).
_STEP_TYPES: dict[str, type[Step]] = {"agent": AgentStep, "tool": ToolStep}

_STEP_SKELETON_FIELDS = frozenset(
    {"source", "messages", "task_call", "error", "started_at", "ended_at", "step_id"}
)


def _step_from_payload(payload: dict[str, Any]) -> Step | None:
    """Reconstruct a Step from a span's hud.payload dict.

    Uses the source-specific subclass when available, falling back to the base
    Step skeleton (source, messages, task_call, error, timing) on any failure.
    """
    cls = _STEP_TYPES.get(payload.get("source", ""), Step)
    with contextlib.suppress(Exception):
        return cls.model_validate(payload)
    skeleton = {k: payload[k] for k in _STEP_SKELETON_FIELDS if k in payload}
    with contextlib.suppress(Exception):
        return Step.model_validate(skeleton)
    return None


def _load_trace_from_local_spans(
    otel_id: str,
) -> tuple[list[Step], str | None] | None:
    """Reconstruct steps and answer from the local JSONL span file for this trace.

    Returns ``(steps, answer)`` when the file exists, ``None`` when it does not.
    Steps are ordered by ``start_time``. The answer is the ``content`` field of
    the last agent step.
    """
    local = Path(settings.span_dir) / f"{otel_id}.jsonl" if settings.span_dir else None
    if local is None or not local.exists():
        return None

    raw_spans: list[dict[str, Any]] = []
    for line in local.read_text(encoding="utf-8").splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):
                raw_spans.append(json.loads(line))

    step_spans = sorted(
        (s for s in raw_spans if s.get("attributes", {}).get("hud.schema") == "hud.step.v1"),
        key=lambda s: s.get("start_time", ""),
    )

    steps = [
        step
        for span in step_spans
        if (step := _step_from_payload(span.get("attributes", {}).get("hud.payload", {})))
        is not None
    ]

    answer: str | None = None
    for span in reversed(step_spans):
        payload = span.get("attributes", {}).get("hud.payload", {})
        if payload.get("source") == "agent":
            raw = payload.get("content")
            answer = raw if isinstance(raw, str) else None
            break

    return steps, answer


def _load_trace_from_platform(
    trace_id: str,
    client: PlatformClient,
) -> tuple[list[Step], str | None]:
    """Fetch steps and answer from the platform spans API.

    NOT YET IMPLEMENTED — this is the integration point for platform-backed
    step replay.

    When ``GET /trace/{id}/spans`` (or equivalent) is available on the platform:

    1. Call the endpoint using ``client.get(f"/trace/{trace_id}/spans")``.
    2. Filter to spans where ``attributes["hud.schema"] == "hud.step.v1"``.
    3. Sort by ``start_time``.
    4. Reconstruct each step via ``_step_from_payload(span["attributes"]["hud.payload"])``.
    5. Extract the answer from the last agent step's ``content`` field.

    Note: ``GET /trace/{id}/events`` is NOT a substitute — it returns rendered
    display events (agent_message / tool_call format), not the raw Step payloads
    needed for typed reconstruction and replay.
    """
    _ = trace_id, client  # available to the future implementation
    raise NotImplementedError(
        "platform span retrieval is not yet implemented; "
        "set HUD_TELEMETRY_LOCAL_DIR so spans are written locally, "
        "then regrade from the local file."
    )


def _build_run(
    trace_id: str,
    task_slug: str,
    client: PlatformClient,
    *,
    group_id: str | None = None,
) -> Run | None:
    """Load spans for one trace and assemble a Run ready for regrading.

    Returns ``None`` and logs a warning when no spans are found or the trace
    has no agent answer (e.g. it errored before the agent responded).
    """
    otel_id = normalize_trace_id(trace_id)
    local_result = _load_trace_from_local_spans(otel_id)
    if local_result is not None:
        steps, answer = local_result
    else:
        try:
            steps, answer = _load_trace_from_platform(trace_id, client)
        except NotImplementedError:
            hud_console.warning(f"  {trace_id[:8]}... skipped — no local spans found")
            return None

    if answer is None:
        hud_console.warning(f"  {trace_id[:8]}... skipped — no agent answer in trace")
        return None

    run = Run(None, "", {})
    run.slug = task_slug
    run.group_id = group_id
    run.trace = Trace(trace_id=otel_id, steps=steps, content=answer)
    return run


def _resolve_placement(
    runtime: str | None,
    source: str,
    taskset: Any,
) -> Provider | HostedRuntime | None:
    if runtime is None:
        return None
    adapter = TypeAdapter(Placement | TcpUrl)
    try:
        parsed = adapter.validate_python(runtime)
    except Exception:
        raise CliError("usage", f"Unknown runtime: {runtime!r}") from None
    match parsed:
        case AnyUrl():
            return Runtime(str(parsed))
        case _:
            return named_placement(parsed, source, taskset)


def _print_job_results(job: Any, elapsed: float) -> None:
    if job.runs and settings.telemetry_enabled and settings.api_key:
        hud_console.info(f"{settings.hud_web_url}/jobs/{UUID(job.id)}")
    for regrade_run in job.runs:
        hud_console.print(
            f"  reward: [green]{regrade_run.reward:.3f}[/green]  "
            f"trace: [blue]{regrade_run.trace_id}[/blue]"
        )
    hud_console.print(f"  time: {elapsed:.1f}s")


@regrade_app.command("trace")
def regrade_trace_command(
    trace_id: str = typer.Argument(..., help="Trace ID of the original run to regrade"),
    source: str = typer.Argument(..., help="Tasks file (.py) or platform taskset name"),
    task_slug: str | None = typer.Option(
        None,
        "--task-slug",
        help="Task slug to regrade. Inferred from the platform if omitted.",
    ),
    runtime: str | None = typer.Option(
        None,
        "--runtime",
        help="Placement: local, hud, hosted, docker, modal, daytona, or a tcp:// url. "
        "Default: local for a tasks file; hosted for a platform taskset.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts."),
) -> dict[str, Any]:
    """Regrade a single completed trace without re-running the agent.

    Provisions the environment, re-runs task setup, replays all original agent
    steps into the new trace, injects the stored answer, and records the new
    grade under a new trace linked to the original via parent_trace_id.

    Requires local span data (HUD_TELEMETRY_LOCAL_DIR must be set, or telemetry
    must be disabled so spans fall back to ~/.hud/spans). Platform span retrieval
    is not yet implemented — see _load_trace_from_platform for the integration
    point once the platform exposes GET /trace/{id}/spans.

    [not dim]Examples:
        hud regrade trace <trace-id> tasks.py
        hud regrade trace <trace-id> tasks.py --task-slug fix_bug-abc123
        hud regrade trace <trace-id> "My Platform Taskset"
        hud regrade trace <trace-id> tasks.py --runtime hosted[/not dim]
    """
    client = PlatformClient.from_settings()

    resolved_slug = task_slug
    resolved_group_id: str | None = None
    if resolved_slug is None:
        with contextlib.suppress(HudRequestError):
            data = client.get(f"/trace/{trace_id}")
            resolved_slug = data.get("task_slug")
            resolved_group_id = data.get("group_id")
    if resolved_slug is None:
        raise CliError(
            "usage",
            f"Cannot determine task slug for trace {trace_id}. Pass --task-slug explicitly.",
            suggestion="Run 'hud jobs get <job-id>' to list traces and their slugs.",
        )

    run = _build_run(trace_id, resolved_slug, client, group_id=resolved_group_id)
    if run is None:
        raise CliError(
            "failure",
            f"Could not load trace {trace_id}.",
            suggestion="Ensure the trace completed successfully and has local span data.",
        )

    taskset = load_taskset(source)
    if not taskset:
        raise CliError(
            "failure",
            f"No runnable tasks found in {source!r}.",
            suggestion="Ensure the source exports Task objects via @env.template.",
        )

    placement = _resolve_placement(runtime, source, taskset)

    answer = run.trace.content or ""
    preview = answer[:60] + ("..." if len(answer) > 60 else "")
    hud_console.info(
        f"Regrading {trace_id[:8]}...  slug={resolved_slug}  "
        f"steps={len(run.trace.steps)}  answer={preview!r}"
    )
    CLI.confirm_or_abort("Proceed?", yes=yes, default=True)

    started = time.monotonic()
    job = asyncio.run(taskset.regrade([run], runtime=placement))
    elapsed = time.monotonic() - started

    _print_job_results(job, elapsed)

    return {
        "job_id": job.id,
        "original_trace_id": trace_id,
        "run_count": len(job.runs),
        "mean_reward": job.reward,
        "error_count": len(job.errors),
        "elapsed_seconds": elapsed,
        "runs": [
            {
                "slug": r.slug,
                "reward": r.reward,
                "is_error": r.trace.is_error,
                "trace_id": r.trace_id,
            }
            for r in job.runs
        ],
    }


@regrade_app.command("job")
def regrade_job_command(
    job_id: str = typer.Argument(..., help="Job ID whose traces to regrade"),
    source: str = typer.Argument(..., help="Tasks file (.py) or platform taskset name"),
    runtime: str | None = typer.Option(
        None,
        "--runtime",
        help="Placement: local, hud, hosted, docker, modal, daytona, or a tcp:// url. "
        "Default: local for a tasks file; hosted for a platform taskset.",
    ),
    limit: int = typer.Option(100, "--limit", "-n", help="Max traces to regrade"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts."),
) -> dict[str, Any]:
    """Regrade all traces in a job without re-running the agent.

    Fetches every trace in the job, loads their stored steps and answers from
    local span files, and regrads them in one batch under a new job linked to
    the original traces via parent_trace_id.

    Traces with no local span data or no agent answer are skipped with a warning.

    [not dim]Examples:
        hud regrade job <job-id> tasks.py
        hud regrade job <job-id> tasks.py --runtime hosted
        hud regrade job <job-id> tasks.py --limit 20[/not dim]
    """
    client = PlatformClient.from_settings()

    try:
        data = client.get(f"/jobs/{UUID(job_id)}/traces", params={"limit": limit})
    except HudRequestError as exc:
        raise CliError.from_http(exc, resource="Job", input={"job_id": job_id}) from exc

    items: list[dict[str, Any]] = data.get("items", [])
    if not items:
        raise CliError("failure", f"No traces found for job {job_id}.")

    hud_console.info(f"Loading {len(items)} trace(s) from job {job_id}...")
    runs: list[Run] = []
    for item in items:
        trace_id = item["id"]
        task_slug = item.get("task_slug")
        if task_slug is None:
            hud_console.warning(f"  {trace_id[:8]}... skipped — no task_slug in job")
            continue
        run = _build_run(trace_id, task_slug, client, group_id=item.get("group_id"))
        if run is not None:
            runs.append(run)

    if not runs:
        raise CliError(
            "failure",
            "No traces could be loaded for regrading.",
            suggestion="Ensure local span data exists (HUD_TELEMETRY_LOCAL_DIR or ~/.hud/spans).",
        )

    taskset = load_taskset(source)
    if not taskset:
        raise CliError(
            "failure",
            f"No runnable tasks found in {source!r}.",
            suggestion="Ensure the source exports Task objects via @env.template.",
        )

    placement = _resolve_placement(runtime, source, taskset)

    skipped = len(items) - len(runs)
    hud_console.info(f"Regrading {len(runs)}/{len(items)} trace(s) (skipped {skipped})")
    CLI.confirm_or_abort("Proceed?", yes=yes, default=True)

    started = time.monotonic()
    job = asyncio.run(taskset.regrade(runs, runtime=placement))
    elapsed = time.monotonic() - started

    _print_job_results(job, elapsed)

    return {
        "job_id": job.id,
        "original_job_id": job_id,
        "run_count": len(job.runs),
        "mean_reward": job.reward,
        "error_count": len(job.errors),
        "skipped_count": len(items) - len(runs),
        "elapsed_seconds": elapsed,
        "runs": [
            {
                "slug": r.slug,
                "reward": r.reward,
                "is_error": r.trace.is_error,
                "trace_id": r.trace_id,
            }
            for r in job.runs
        ],
    }
