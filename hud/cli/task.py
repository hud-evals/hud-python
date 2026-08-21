"""``hud task`` — start a task (get its prompt) or grade an answer.

Placement-explicit: the source flow spawns the env source on a local substrate
(the same ``spawn`` provider ``hud eval`` uses) and speaks the protocol to it;
``--url`` attaches to an already-served control channel instead.

    hud task list                          # what tasks this source exposes
    hud task start fix_config              # -> the task's prompt (stdout)
    hud task grade fix_config --answer "…" # -> the reward (stdout); --out for JSON
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - Typer resolves annotations at runtime
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import typer

from hud.cli.task_runtime import (
    TaskResolutionError,
    collect_taskset,
    find_local_env_url,
    normalize_control_url,
    parse_task_args,
    select_local_task,
    spawn_target,
)
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from hud.clients import HudClient
    from hud.eval import Taskset
    from hud.eval.runtime import Runtime

hud_console = HUDConsole()

task_app = typer.Typer(
    help="Start a task or grade an answer (attaches to a running env, or spawns from source).",
    rich_markup_mode="rich",
)


PhaseStatus = Literal["pass", "fail", "skip"]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class CheckPhase:
    name: str
    status: PhaseStatus
    detail: str


def _resolution_or_exit(operation: Callable[[], T]) -> T:
    try:
        return operation()
    except TaskResolutionError as exc:
        message = str(exc)
    hud_console.error(message)
    raise typer.Exit(1)


def _parse_args(args: str) -> dict[str, Any]:
    return _resolution_or_exit(lambda: parse_task_args(args))


def _collect(source: str) -> Taskset:
    """Collect a Taskset from a source (``.py``/dir or JSON/JSONL), like ``hud eval``."""
    return _resolution_or_exit(lambda: collect_taskset(source))


def _environment_name(value: str) -> str:
    return value.removeprefix("env/").removeprefix("environment/")


def _task_id(value: str) -> str:
    return value.removeprefix("task/")


def _resolve(
    task: str,
    source: str | None,
    url: str | None,
    env: str | None,
    args: dict[str, Any],
) -> tuple[str, dict[str, Any], AbstractAsyncContextManager[Runtime]]:
    """Resolve ``(task_id, args, placement)``, choosing a substrate in priority order:

    1. ``--env`` — boot that deployed environment through the HUD runtime;
    2. ``--url`` — attach to that control channel;
    3. no ``--source`` and a local env already serving on :8765 — attach to it
       (e.g. inside a built image, or alongside ``hud serve``);
    4. otherwise — introspect local source for the task id/slug, and spawn that
       source as the substrate.

    The placement decision is made *here*, so this returns the acquisition
    itself (one substrate, ready to enter), not a provider. ``--args`` (when
    given) overrides the authored args so any explicit parameterization is
    runnable.
    """
    from contextlib import nullcontext

    from hud.eval import HUDRuntime, Task
    from hud.eval.runtime import Runtime, SubprocessRuntime

    if sum(value is not None for value in (source, url, env)) > 1:
        hud_console.error("choose only one placement: --source, --url, or --env")
        raise typer.Exit(1)
    task_id = _task_id(task)
    if env is not None:
        selected = Task(env=_environment_name(env), id=task_id, args=args)
        return selected.id, selected.args, HUDRuntime()(selected)

    attach = url
    if attach is None and source is None:
        attach = find_local_env_url()
    if attach is not None:
        endpoint = _resolution_or_exit(lambda: normalize_control_url(attach))
        return task_id, args, nullcontext(Runtime(endpoint))

    selected = _resolution_or_exit(lambda: select_local_task(task_id, source or ".", args))
    placement = SubprocessRuntime(spawn_target(source or "."))(selected)
    return selected.id, selected.args, placement


def _emit(result: dict[str, Any], headline: str, out: Path | None) -> None:
    """Thin output: the full protocol frame to ``--out``, else the headline value to stdout."""
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return
    value = result.get(headline, result)
    typer.echo(value if isinstance(value, str) else json.dumps(value, default=str))


async def _require_task(client: HudClient, task_id: str) -> None:
    tasks = await client.list_tasks()
    available = [
        task["id"] for task in tasks if isinstance(task, dict) and isinstance(task.get("id"), str)
    ]
    if task_id not in available:
        joined = ", ".join(available) or "<none>"
        raise TaskResolutionError(f"task {task_id!r} is not exposed by the environment ({joined})")


def _count_subscores(subscores: object) -> int:
    if subscores is None:
        return 0
    if not isinstance(subscores, list):
        raise ValueError("grade subscores must be a list")

    count = 0
    for subscore in subscores:
        if not isinstance(subscore, dict):
            raise ValueError("each grade subscore must be an object")
        count += 1 + _count_subscores(subscore.get("children"))
    return count


def _validate_grade_result(result: dict[str, Any]) -> tuple[float, int]:
    from hud.clients.client import HudProtocolError
    from hud.eval.run import Grade

    try:
        grade = Grade.from_dict(result)
    except (HudProtocolError, ValueError) as exc:
        raise ValueError(str(exc)) from None
    if grade.is_error:
        raise ValueError(grade.content or "grader returned isError=true")
    score = grade.reward
    if not math.isfinite(score):
        raise ValueError(f"grade score must be finite, got {score}")
    if not 0 <= score <= 1:
        raise ValueError(f"grade score must be finite and within [0, 1], got {score}")
    return score, _count_subscores(grade.raw.get("subscores"))


def _render_check(phases: list[CheckPhase]) -> None:
    for phase in phases:
        typer.echo(f"[{phase.status}] {phase.name:<8} {phase.detail}")
    passed = all(phase.status != "fail" for phase in phases)
    typer.echo(f"\nresult: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise typer.Exit(1)


def _phase_error(exc: Exception, action: str, timeout: float) -> str:
    return f"{action} timed out after {timeout:g}s" if isinstance(exc, TimeoutError) else str(exc)


async def _dry_run_grade(
    task_id: str,
    task_args: dict[str, Any],
    placement: AbstractAsyncContextManager[Runtime],
    phase_timeout: float,
) -> tuple[list[CheckPhase], dict[str, Any] | None]:
    phases: list[CheckPhase] = []
    from hud.clients import connect

    async with contextlib.AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(phase_timeout):
                runtime = await stack.enter_async_context(placement)
                client = await stack.enter_async_context(
                    connect(runtime, ready_timeout=phase_timeout)
                )
        except Exception as exc:
            phases.append(
                CheckPhase(
                    "env",
                    "fail",
                    _phase_error(exc, "environment startup", phase_timeout),
                )
            )
            phases.extend(
                [
                    CheckPhase("task", "skip", "environment check failed"),
                    CheckPhase("grader", "skip", "task did not start"),
                    CheckPhase("reward", "skip", "grader did not run"),
                ]
            )
            return phases, None

        try:
            tasks = await asyncio.wait_for(client.list_tasks(), phase_timeout)
        except Exception as exc:
            phases.append(CheckPhase("env", "fail", _phase_error(exc, "tasks.list", phase_timeout)))
            phases.extend(
                [
                    CheckPhase("task", "skip", "environment check failed"),
                    CheckPhase("grader", "skip", "task did not start"),
                    CheckPhase("reward", "skip", "grader did not run"),
                ]
            )
            return phases, None

        manifest = client.manifest
        env_name = manifest.server_info.name if manifest is not None else "environment"
        phases.append(CheckPhase("env", "pass", f"{env_name} ready, {len(tasks)} task(s)"))
        available = {
            task["id"]
            for task in tasks
            if isinstance(task, dict) and isinstance(task.get("id"), str)
        }
        if task_id not in available:
            phases.extend(
                [
                    CheckPhase("task", "fail", f"task {task_id!r} is not exposed"),
                    CheckPhase("grader", "skip", "task did not start"),
                    CheckPhase("reward", "skip", "grader did not run"),
                ]
            )
            return phases, None

        session_active = False
        try:
            try:
                session_active = True
                started = await asyncio.wait_for(
                    client.start_task(task_id, task_args), phase_timeout
                )
            except Exception as exc:
                phases.append(
                    CheckPhase(
                        "task",
                        "fail",
                        _phase_error(exc, "tasks.start", phase_timeout),
                    )
                )
                phases.extend(
                    [
                        CheckPhase("grader", "skip", "task did not start"),
                        CheckPhase("reward", "skip", "grader did not run"),
                    ]
                )
                return phases, None

            prompt = started.get("prompt")
            detail = (
                f"start returned prompt ({len(prompt)} chars)"
                if isinstance(prompt, str)
                else "start returned a prompt"
            )
            phases.append(CheckPhase("task", "pass", detail))

            try:
                graded = await asyncio.wait_for(client.grade({"answer": ""}), phase_timeout)
                session_active = False
            except Exception as exc:
                phases.append(
                    CheckPhase(
                        "grader",
                        "fail",
                        _phase_error(exc, "tasks.grade", phase_timeout),
                    )
                )
                phases.append(CheckPhase("reward", "skip", "grader failed"))
                return phases, None

            if graded.get("isError") is True:
                detail = str(graded.get("content") or "grader returned isError=true")
                phases.append(CheckPhase("grader", "fail", detail))
                phases.append(CheckPhase("reward", "skip", "grader failed"))
                return phases, graded
            phases.append(CheckPhase("grader", "pass", "empty-answer grade completed"))
            try:
                score, subscore_count = _validate_grade_result(graded)
            except ValueError as exc:
                phases.append(CheckPhase("reward", "fail", str(exc)))
                return phases, graded

            detail = f"score {score:g} is valid"
            if subscore_count:
                detail += f", {subscore_count} subscore(s) valid"
            phases.append(CheckPhase("reward", "pass", detail))
            return phases, graded
        finally:
            if session_active:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(client.cancel(), 2.0)


@task_app.command("list")
def list_command(
    source: str = typer.Option(".", "--source", "-s", help="Env source (.py/dir/JSON)."),
    env: str | None = typer.Option(
        None,
        "--env",
        help="Boot this deployed environment and list its live task manifest.",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        "-u",
        help="List tasks from a served control channel instead of local source.",
    ),
) -> None:
    """List tasks from local definitions or a live environment."""
    if env is not None and url is not None:
        hud_console.error("choose either --env or --url")
        raise typer.Exit(1)
    if env is not None or url is not None:
        from hud.eval import HUDRuntime, Runtime, Task

        env_name = _environment_name(env) if env else "attached"
        task = Task(env=env_name, id="__hud_task_list__")
        provider = (
            HUDRuntime()
            if env is not None
            else Runtime(_resolution_or_exit(lambda: normalize_control_url(url or "")))
        )

        async def _run() -> list[dict[str, Any]]:
            from hud.clients import connect

            async with provider(task) as runtime, connect(runtime) as client:
                return await client.list_tasks()

        for task_manifest in asyncio.run(_run()):
            typer.echo(
                f"{task_manifest.get('id', '<unknown>')}\t"
                f"{task_manifest.get('description', '')}".rstrip()
            )
        return

    for slug, task in _collect(source).items():
        args = f" {json.dumps(task.args)}" if task.args else ""
        typer.echo(f"{slug}\t{task.id}{args}")


@task_app.command("start")
def start_command(
    task: str = typer.Argument(..., help="Task id or slug."),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Spawn this env source (.py/dir/JSON) instead of attaching."
    ),
    args: str = typer.Option("{}", "--args", "-a", help="JSON object of task args."),
    url: str | None = typer.Option(
        None, "--url", "-u", help="Attach to a served control channel instead of loading source."
    ),
    env: str | None = typer.Option(
        None,
        "--env",
        help="Boot this deployed environment instead of loading local source.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="Write the prompt here instead of stdout."
    ),
) -> None:
    """Start a task and return its prompt (the env's first yield)."""
    task_id, task_args, placement = _resolve(task, source, url, env, _parse_args(args))

    async def _run() -> dict[str, Any]:
        from hud.clients import connect

        # Start and disconnect without grading; an attached (persistent) env keeps
        # the session for a later `hud task grade` to resume.
        async with placement as runtime, connect(runtime) as client:
            await _require_task(client, task_id)
            return await client.start_task(task_id, task_args)

    result = _resolution_or_exit(lambda: asyncio.run(_run()))
    _emit(result, "prompt", out)


@task_app.command("grade")
def grade_command(
    task: str = typer.Argument(..., help="Task id or slug."),
    answer: str = typer.Option("", "--answer", help="Answer to grade."),
    answer_file: Path | None = typer.Option(  # noqa: B008
        None, "--answer-file", help="Read the answer from a file instead of --answer."
    ),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Spawn this env source (.py/dir/JSON) instead of attaching."
    ),
    args: str = typer.Option("{}", "--args", "-a", help="JSON object of task args."),
    url: str | None = typer.Option(
        None, "--url", "-u", help="Attach to a served control channel instead of loading source."
    ),
    env: str | None = typer.Option(
        None,
        "--env",
        help="Boot this deployed environment instead of loading local source.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="Write the full JSON result here (else print the reward)."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Start fresh, grade an empty answer, and report deterministic lifecycle checks.",
    ),
    timeout: float = typer.Option(
        120.0,
        "--timeout",
        min=0.1,
        help="Per-phase timeout in seconds for --dry-run.",
    ),
) -> None:
    """Grade an answer for a task and return its reward."""
    if dry_run and (answer or answer_file is not None):
        hud_console.error("--dry-run uses an empty answer; omit --answer and --answer-file")
        raise typer.Exit(1)
    answer_text = answer_file.read_text(encoding="utf-8") if answer_file is not None else answer
    task_id, task_args, placement = _resolve(task, source, url, env, _parse_args(args))

    if dry_run:
        phases, result = asyncio.run(_dry_run_grade(task_id, task_args, placement, timeout))
        if out is not None and result is not None:
            _emit(result, "score", out)
        _render_check(phases)
        return

    async def _run() -> dict[str, Any]:
        from hud.clients import connect
        from hud.clients.client import HudProtocolError

        async with placement as runtime, connect(runtime) as client:
            session_active = False
            try:
                try:
                    result = await client.grade({"answer": answer_text})  # resume a prior start
                except HudProtocolError as exc:
                    if exc.code != -32600 or exc.message != "no task in progress":
                        raise
                    await _require_task(client, task_id)
                    session_active = True
                    await client.start_task(task_id, task_args)
                    result = await client.grade({"answer": answer_text})
                    session_active = False
                return result
            finally:
                if session_active:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(client.cancel(), 2.0)

    result = _resolution_or_exit(lambda: asyncio.run(_run()))
    _emit(result, "score", out)


__all__ = ["task_app"]
