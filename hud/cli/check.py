"""Check a HUD Task through the existing lifecycle primitives."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - Typer resolves annotations at runtime
from typing import Any, Literal

import typer
from pydantic import BaseModel, ConfigDict, Field

from hud.cli.task_runtime import (
    TaskResolutionError,
    attached_task,
    find_local_env_url,
    normalize_control_url,
    parse_task_args,
    select_local_task,
    spawn_target,
)

CheckStatus = Literal["passed", "failed", "error", "skipped"]
CheckOutcome = Literal["passed", "failed", "error"]
CheckMode = Literal["oracle", "agent", "start-only"]
ErrorKind = Literal["input", "execution"]

_CRITERIA = (
    "resolution",
    "environment_startup",
    "task_startup",
    "grader_execution",
    "oracle_or_agent_reward",
)
_SECRET_MARKERS = ("api_key", "apikey", "authorization", "cookie", "password", "secret", "token")


class CheckCriterion(BaseModel):
    """One stable Task lifecycle criterion."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: CheckStatus
    detail: str
    evidence: dict[str, Any] | None = None


class TaskCheckReport(BaseModel):
    """Versioned output contract for ``hud check``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["hud.task-check.v1"]
    outcome: CheckOutcome
    mode: CheckMode
    task_id: str
    runtime: str
    reward: float | None = None
    min_reward: float
    trace_id: str | None = None
    criteria: list[CheckCriterion]
    error: str | None = None
    error_kind: ErrorKind | None = None
    duration_seconds: float = Field(ge=0)


@dataclass(slots=True)
class CheckRequest:
    """Validated command inputs passed into the asynchronous checker."""

    task: str
    source: str | None = None
    args_json: str = "{}"
    url: str | None = None
    runtime: str | None = None
    remote: bool = False
    answer: str | None = None
    agent: str | None = None
    model: str | None = None
    start_only: bool = False
    min_reward: float = 1.0
    timeout: float = 3600.0
    startup_timeout: float = 120.0
    max_steps: int = 10
    gateway: bool = False
    config: list[str] = field(default_factory=list)

    @property
    def mode(self) -> CheckMode:
        if self.start_only:
            return "start-only"
        return "agent" if self.agent is not None else "oracle"


def _criteria_template() -> dict[str, CheckCriterion]:
    return {
        name: CheckCriterion(name=name, status="skipped", detail="not reached")
        for name in _CRITERIA
    }


def _safe_value(value: Any, *, string_limit: int = 2_000) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(marker in str(key).lower() for marker in _SECRET_MARKERS)
                else _safe_value(item, string_limit=string_limit)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, string_limit=string_limit) for item in value[:50]]
    if isinstance(value, str) and len(value) > string_limit:
        return f"{value[:string_limit]}…[truncated]"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _redact_evidence(value: Any, *, max_chars: int = 4_000) -> dict[str, Any]:
    """Return bounded JSON evidence with common credential fields removed."""
    safe = _safe_value(value)
    if not isinstance(safe, dict):
        safe = {"value": safe}
    encoded = json.dumps(safe, sort_keys=True, default=str)
    if len(encoded) <= max_chars:
        return safe
    preview_limit = max(0, max_chars - 50)
    return {"truncated": True, "preview": encoded[:preview_limit]}


def _runtime_label(request: CheckRequest, attached_url: str | None) -> str:
    if request.remote:
        return "hosted"
    if request.runtime == "hud":
        return "hud"
    if attached_url is not None:
        return attached_url
    return "local"


def _resolve(request: CheckRequest) -> tuple[Any, Any, str]:
    """Resolve one Task row and its existing runtime provider."""
    from hud.eval import HostedRuntime, HUDRuntime, Runtime, SubprocessRuntime
    from hud.settings import settings

    if (request.remote or request.runtime == "hud" or request.gateway) and not settings.api_key:
        raise TaskResolutionError(
            "HUD_API_KEY is required for HUD runtime, hosted, or gateway checks",
        )

    args = parse_task_args(request.args_json)
    attached_url = request.url
    if (
        attached_url is None
        and request.source is None
        and request.runtime is None
        and not request.remote
    ):
        attached_url = find_local_env_url()

    if request.source is not None:
        task, source_path = select_local_task(request.task, request.source, args)
    elif attached_url is not None:
        task, source_path = attached_task(request.task, args), None
    else:
        task, source_path = select_local_task(request.task, ".", args)

    if request.remote:
        provider = HostedRuntime(run_timeout=request.timeout)
    elif request.runtime == "hud":
        provider = HUDRuntime(run_timeout=request.timeout)
    elif attached_url is not None:
        attached_url = normalize_control_url(attached_url)
        provider = Runtime(attached_url)
    else:
        assert source_path is not None
        provider = SubprocessRuntime(
            spawn_target(source_path),
            ready_timeout=request.startup_timeout,
        )
    return task, provider, _runtime_label(request, attached_url)


def _agent(request: CheckRequest) -> Any:
    from hud.cli.eval import EvalConfig, _build_agent

    try:
        config = EvalConfig().merge_cli(
            source=request.source,
            agent=request.agent,
            model=request.model,
            max_steps=request.max_steps,
            gateway=request.gateway,
            config=request.config,
            runtime=request.runtime,
            remote=request.remote,
        )
        config.validate_api_keys()
        return _build_agent(config)
    except typer.Exit:
        raise TaskResolutionError("agent credentials or model configuration are invalid") from None
    except (RuntimeError, ValueError) as exc:
        raise TaskResolutionError(str(exc)) from None


async def _run_direct(
    request: CheckRequest,
    task: Any,
    provider: Any,
    criteria: dict[str, CheckCriterion],
) -> tuple[float | None, str | None]:
    """Run start/grade directly against the control channel."""
    from hud.clients import connect

    async with contextlib.AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(request.startup_timeout):
                runtime = await stack.enter_async_context(provider(task))
                client = await stack.enter_async_context(connect(runtime))
        except TimeoutError as exc:
            detail = f"environment did not become ready within {request.startup_timeout:g}s"
            criteria["environment_startup"] = CheckCriterion(
                name="environment_startup",
                status="error",
                detail=detail,
            )
            raise TimeoutError(detail) from exc
        except Exception as exc:
            criteria["environment_startup"] = CheckCriterion(
                name="environment_startup",
                status="error",
                detail=f"environment did not become ready: {exc}",
            )
            raise
        criteria["environment_startup"] = CheckCriterion(
            name="environment_startup",
            status="passed",
            detail="environment control channel is ready",
            evidence=_redact_evidence({"runtime_url": runtime.url}),
        )

        session_active = False
        phase = "task_startup"
        try:
            started = await client.start_task(task.id, task.args)
            session_active = True
            criteria["task_startup"] = CheckCriterion(
                name="task_startup",
                status="passed",
                detail="task started successfully",
                evidence=_redact_evidence(started),
            )

            if request.start_only:
                criteria["grader_execution"] = CheckCriterion(
                    name="grader_execution",
                    status="skipped",
                    detail="explicit --start-only check",
                )
                criteria["oracle_or_agent_reward"] = CheckCriterion(
                    name="oracle_or_agent_reward",
                    status="skipped",
                    detail="explicit --start-only check",
                )
                return None, None

            assert request.answer is not None
            phase = "grader_execution"
            graded = await client.grade({"answer": request.answer})
            session_active = False
            raw_score = graded["score"]
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise TypeError("grade score is not numeric")
            reward = float(raw_score)
            if not math.isfinite(reward):
                raise ValueError("grade score is not finite")
            criteria["grader_execution"] = CheckCriterion(
                name="grader_execution",
                status="passed",
                detail="grader returned a reward",
                evidence=_redact_evidence(graded),
            )
            return reward, None
        except BaseException as exc:
            criteria[phase] = CheckCriterion(
                name=phase,
                status="error",
                detail=(
                    f"task did not start: {exc}"
                    if phase == "task_startup"
                    else f"grader did not return a numeric reward: {exc}"
                ),
            )
            raise
        finally:
            if session_active:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.cancel(), timeout=2.0)


def _run_error(run: Any) -> str | None:
    if run.trace.status == "cancelled":
        return run.trace.error or "rollout was cancelled"
    if run.trace.status != "error" and not run.grade.is_error:
        return None
    return run.trace.error or run.grade.content or "rollout returned an error"


def _error_phase(detail: str) -> str:
    lowered = detail.lower()
    timeout_phase = lowered.rsplit(" during ", 1)[-1] if " during " in lowered else None
    if (
        "[provisioning]" in lowered
        or "[connecting]" in lowered
        or timeout_phase in {"provisioning", "connecting"}
    ):
        return "environment_startup"
    if "[starting task]" in lowered or timeout_phase == "starting task":
        return "task_startup"
    if (
        "[grading]" in lowered
        or "[verifying]" in lowered
        or timeout_phase
        in {
            "grading",
            "verifying",
            "provisioning verifier",
            "cleanup",
        }
    ):
        return "grader_execution"
    return "oracle_or_agent_reward"


async def _run_agent(
    request: CheckRequest,
    task: Any,
    provider: Any,
    agent: Any,
    criteria: dict[str, CheckCriterion],
) -> tuple[float | None, str | None]:
    """Run a full agent rollout locally, through HUD, or hosted."""
    from hud.eval import HostedRuntime
    from hud.eval.run import rollout

    if isinstance(provider, HostedRuntime):
        job = await task.run(
            agent,
            runtime=provider,
            rollout_timeout=request.timeout,
        )
        if not job.runs:
            raise RuntimeError("hosted rollout returned no run")
        run = job.runs[0]
    else:
        run = await rollout(
            task,
            agent,
            runtime=provider,
            rollout_timeout=request.timeout,
        )

    detail = _run_error(run)
    if detail is not None:
        phase = _error_phase(detail)
        reached = True
        for name in _CRITERIA[1:4]:
            if name == phase:
                criteria[name] = CheckCriterion(name=name, status="error", detail=detail)
                reached = False
            elif reached:
                criteria[name] = CheckCriterion(
                    name=name,
                    status="passed",
                    detail=f"{name.replace('_', ' ')} completed",
                )
        if phase == "oracle_or_agent_reward":
            criteria[phase] = CheckCriterion(name=phase, status="error", detail=detail)
        return None, run.trace.trace_id

    for name in _CRITERIA[1:4]:
        criteria[name] = CheckCriterion(
            name=name,
            status="passed",
            detail=f"{name.replace('_', ' ')} completed",
        )
    return float(run.reward), run.trace.trace_id


def _input_error_report(
    request: CheckRequest,
    *,
    started_at: float,
    criteria: dict[str, CheckCriterion],
    detail: str,
) -> TaskCheckReport:
    criteria["resolution"] = CheckCriterion(
        name="resolution",
        status="error",
        detail=detail,
    )
    return TaskCheckReport(
        schema_version="hud.task-check.v1",
        outcome="error",
        mode=request.mode,
        task_id=request.task,
        runtime=_runtime_label(request, request.url),
        min_reward=request.min_reward,
        criteria=list(criteria.values()),
        error=detail,
        error_kind="input",
        duration_seconds=time.monotonic() - started_at,
    )


async def _run_check(request: CheckRequest) -> TaskCheckReport:
    started_at = time.monotonic()
    criteria = _criteria_template()
    agent_instance: Any = None
    try:
        task, provider, runtime_label = _resolve(request)
        if request.mode == "agent":
            agent_instance = _agent(request)
    except TaskResolutionError as exc:
        return _input_error_report(
            request,
            started_at=started_at,
            criteria=criteria,
            detail=str(exc),
        )

    criteria["resolution"] = CheckCriterion(
        name="resolution",
        status="passed",
        detail="task and placement resolved",
        evidence=_redact_evidence({"task_id": task.id, "environment": task.env}),
    )
    reward: float | None = None
    trace_id: str | None = None
    error: str | None = None
    try:
        if request.mode == "agent":
            reward, trace_id = await _run_agent(
                request,
                task,
                provider,
                agent_instance,
                criteria,
            )
            error = next(
                (
                    criterion.detail
                    for criterion in criteria.values()
                    if criterion.status == "error"
                ),
                None,
            )
        else:
            async with asyncio.timeout(request.timeout):
                reward, trace_id = await _run_direct(request, task, provider, criteria)
    except asyncio.CancelledError:
        error = "check cancelled"
    except TimeoutError as exc:
        error = str(exc) or f"check did not complete within {request.timeout:g}s"
    except Exception as exc:
        error = str(exc) or type(exc).__name__

    if error is not None:
        if not any(criterion.status == "error" for criterion in criteria.values()):
            failed_name = next(
                (name for name in _CRITERIA[1:] if criteria[name].status == "skipped"),
                "oracle_or_agent_reward",
            )
            criteria[failed_name] = CheckCriterion(
                name=failed_name,
                status="error",
                detail=error,
            )
        outcome: CheckOutcome = "error"
        error_kind: ErrorKind | None = "execution"
    elif request.start_only:
        outcome, error_kind = "passed", None
    else:
        assert reward is not None
        passed = reward >= request.min_reward
        criteria["oracle_or_agent_reward"] = CheckCriterion(
            name="oracle_or_agent_reward",
            status="passed" if passed else "failed",
            detail=(
                f"reward {reward:g} meets minimum {request.min_reward:g}"
                if passed
                else f"reward {reward:g} is below minimum {request.min_reward:g}"
            ),
            evidence={"reward": reward, "min_reward": request.min_reward},
        )
        outcome, error_kind = ("passed", None) if passed else ("failed", None)

    return TaskCheckReport(
        schema_version="hud.task-check.v1",
        outcome=outcome,
        mode=request.mode,
        task_id=task.id,
        runtime=runtime_label,
        reward=reward,
        min_reward=request.min_reward,
        trace_id=trace_id,
        criteria=list(criteria.values()),
        error=error,
        error_kind=error_kind,
        duration_seconds=time.monotonic() - started_at,
    )


def _print_report(report: TaskCheckReport, *, as_json: bool) -> None:
    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return
    typer.echo(f"HUD Task Check: {report.outcome.upper()}")
    typer.echo(f"Task: {report.task_id}")
    typer.echo(f"Mode: {report.mode}  Runtime: {report.runtime}")
    for criterion in report.criteria:
        marker = {"passed": "PASS", "failed": "FAIL", "error": "ERROR", "skipped": "SKIP"}[
            criterion.status
        ]
        typer.echo(f"[{marker}] {criterion.name}: {criterion.detail}")
    if report.reward is not None:
        typer.echo(f"Reward: {report.reward:g} (minimum {report.min_reward:g})")
    if report.trace_id is not None:
        typer.echo(f"Trace: {report.trace_id}")
    if report.error is not None:
        typer.echo(f"Error: {report.error}", err=True)


def _exit_code(report: TaskCheckReport) -> int:
    if report.error_kind == "input":
        return 2
    return {"passed": 0, "failed": 1, "error": 3}[report.outcome]


def check_command(
    task: str = typer.Argument(..., help="Task id or local Task slug."),
    source: str | None = typer.Option(None, "--source", "-s", help="Task/env source path."),
    args: str = typer.Option("{}", "--args", help="Task arguments as a JSON object."),
    url: str | None = typer.Option(None, "--url", help="Attach to a tcp:// control channel."),
    runtime: str | None = typer.Option(None, "--runtime", help="Placement: local or hud."),
    remote: bool = typer.Option(False, "--remote", help="Run the agent rollout on HUD."),
    answer: str | None = typer.Option(None, "--answer", help="Direct oracle answer."),
    answer_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--answer-file",
        exists=False,
        dir_okay=False,
        help="Read the direct oracle answer from a file.",
    ),
    agent: str | None = typer.Option(None, "--agent", help="Gateway agent type."),
    model: str | None = typer.Option(None, "--model", help="Agent model override."),
    start_only: bool = typer.Option(False, "--start-only", help="Only verify Task startup."),
    min_reward: float = typer.Option(1.0, "--min-reward", min=0.0),
    timeout: float = typer.Option(3600.0, "--timeout", min=0.1),
    startup_timeout: float = typer.Option(120.0, "--startup-timeout", min=0.1),
    max_steps: int = typer.Option(10, "--max-steps", min=1),
    gateway: bool = typer.Option(False, "--gateway", help="Route local agent calls through HUD."),
    config: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--config",
        help="Agent KEY=VALUE override.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit hud.task-check.v1 JSON."),
) -> None:
    """Check Task resolution, startup, grading, and reward."""
    strategies = int(answer is not None) + int(answer_file is not None) + int(agent is not None)
    strategies += int(start_only)
    if strategies != 1:
        typer.echo(
            "Error: choose exactly one proof strategy: --answer/--answer-file, "
            "--agent, or --start-only.",
            err=True,
        )
        raise typer.Exit(2)
    if remote and agent is None:
        typer.echo("Error: --remote requires --agent.", err=True)
        raise typer.Exit(2)
    if sum((url is not None, runtime is not None, remote)) > 1:
        typer.echo("Error: choose only one placement: --url, --runtime, or --remote.", err=True)
        raise typer.Exit(2)
    if runtime not in (None, "local", "hud"):
        typer.echo("Error: --runtime must be local or hud.", err=True)
        raise typer.Exit(2)
    agent_only_options = model is not None or gateway or bool(config) or max_steps != 10
    if agent is None and agent_only_options:
        typer.echo(
            "Error: --model, --gateway, --config, and --max-steps require --agent.",
            err=True,
        )
        raise typer.Exit(2)

    if answer_file is not None:
        try:
            answer = answer_file.read_text(encoding="utf-8")
        except OSError as exc:
            typer.echo(f"Error: cannot read --answer-file: {exc}", err=True)
            raise typer.Exit(2) from None

    request = CheckRequest(
        task=task,
        source=source,
        args_json=args,
        url=url,
        runtime=runtime,
        remote=remote,
        answer=answer,
        agent=agent,
        model=model,
        start_only=start_only,
        min_reward=min_reward,
        timeout=timeout,
        startup_timeout=startup_timeout,
        max_steps=max_steps,
        gateway=gateway,
        config=config or [],
    )
    try:
        report = asyncio.run(_run_check(request))
    except KeyboardInterrupt:
        criteria = _criteria_template()
        criteria["environment_startup"] = CheckCriterion(
            name="environment_startup",
            status="error",
            detail="check cancelled",
        )
        report = TaskCheckReport(
            schema_version="hud.task-check.v1",
            outcome="error",
            mode=request.mode,
            task_id=request.task,
            runtime=_runtime_label(request, request.url),
            min_reward=request.min_reward,
            criteria=list(criteria.values()),
            error="check cancelled",
            error_kind="execution",
            duration_seconds=0,
        )
    _print_report(report, as_json=json_output)
    raise typer.Exit(_exit_code(report))


__all__ = [
    "CheckCriterion",
    "CheckRequest",
    "TaskCheckReport",
    "check_command",
]
