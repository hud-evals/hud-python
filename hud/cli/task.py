"""``hud task`` — start a task (get its prompt) or grade an answer.

The task source resolves an authored slug to its template id and bound args.
Without ``--url`` that source is also spawned locally; with ``--url`` the task
runs against the already-served control channel instead.

    hud task list                          # what tasks this source exposes
    hud task start fix_config              # -> the task's prompt (stdout)
    hud task grade fix_config --answer "…" # -> the reward (stdout); --out for JSON
"""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import typer

from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from hud.eval.runtime import Runtime

hud_console = HUDConsole()

task_app = typer.Typer(
    help="Start a task or grade an answer (attaches to a running env, or spawns from source).",
    rich_markup_mode="rich",
)


def _parse_args(args: str) -> dict[str, Any]:
    try:
        parsed = json.loads(args or "{}")
    except json.JSONDecodeError as exc:
        hud_console.error(f"--args must be valid JSON: {exc}")
        raise typer.Exit(1) from None
    if not isinstance(parsed, dict):
        hud_console.error("--args must be a JSON object")
        raise typer.Exit(1)
    return parsed


def _collect(source: str) -> Any:
    """Collect a Taskset from a source (``.py``/dir or JSON/JSONL), like ``hud eval``."""
    from hud.eval import Taskset

    try:
        return Taskset.from_file(source)
    except FileNotFoundError as exc:
        hud_console.error(str(exc))
        raise typer.Exit(1) from None


def _local_env_url(port: int = 8765) -> str | None:
    """Return a control-channel URL if an env is already serving locally on ``port``
    (e.g. ``hud serve``, or a built image whose CMD serves on :8765), else ``None``."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return f"tcp://127.0.0.1:{port}"
    except OSError:
        return None


def _spawn_target(source: str) -> Path:
    """The path ``spawn`` serves: ``.py``/dir as-is, JSON/JSONL's parent directory."""
    resolved = Path(source).resolve()
    if resolved.is_dir() or resolved.suffix == ".py":
        return resolved
    return resolved.parent


def _resolve(
    task: str, source: str | None, url: str | None, args: dict[str, Any]
) -> tuple[str, dict[str, Any], AbstractAsyncContextManager[Runtime]]:
    """Resolve ``(task_id, args, placement)``.

    ``--source`` resolves an authored task id/slug and its bound args. ``--url``
    selects an existing substrate; otherwise an explicit source is spawned. With
    neither option, a local env on :8765 is used when present, or ``.`` is resolved
    and spawned.

    ``--args`` overrides authored args when supplied.
    """
    from contextlib import nullcontext

    from hud.eval.runtime import Runtime, SubprocessRuntime

    attach = url
    if attach is None and source is None:
        attach = _local_env_url()
    endpoint = None
    if attach is not None:
        parts = urlsplit(attach if "://" in attach else f"tcp://{attach}")
        endpoint = f"tcp://{parts.hostname or '127.0.0.1'}:{parts.port or 8765}"

    if endpoint is not None and source is None:
        return task, args, nullcontext(Runtime(endpoint))

    taskset = _collect(source or ".")
    if not taskset:
        hud_console.error(f"No tasks found in {source or '.'}")
        raise typer.Exit(1)
    matches = [
        candidate
        for index, (slug, candidate) in enumerate(taskset.items())
        if task in (slug, candidate.id, str(index))
    ]
    if not matches:
        available = ", ".join(sorted({t.id for t in taskset}))
        hud_console.error(f"No task matching {task!r} (available: {available})")
        raise typer.Exit(1)
    selected = matches[0]
    if endpoint is not None:
        placement = nullcontext(Runtime(endpoint))
    else:
        placement = SubprocessRuntime(_spawn_target(source or "."))(selected)
    return selected.id, args or selected.args, placement


def _emit(result: dict[str, Any], headline: str, out: Path | None) -> None:
    """Thin output: the full protocol frame to ``--out``, else the headline value to stdout."""
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return
    value = result.get(headline, result)
    typer.echo(value if isinstance(value, str) else json.dumps(value, default=str))


@task_app.command("list")
def list_command(
    source: str = typer.Option(".", "--source", "-s", help="Env source (.py/dir/JSON)."),
) -> None:
    """List the tasks (slug + task id + args) exposed by a source."""
    for slug, task in _collect(source).items():
        args = f" {json.dumps(task.args)}" if task.args else ""
        typer.echo(f"{slug}\t{task.id}{args}")


@task_app.command("start")
def start_command(
    task: str = typer.Argument(..., help="Task id or slug."),
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Resolve the task from this source (.py/dir/JSON); spawn it unless --url is set.",
    ),
    args: str = typer.Option("{}", "--args", "-a", help="JSON object of task args."),
    url: str | None = typer.Option(
        None,
        "--url",
        "-u",
        help="Run against this served control channel; --source may still resolve the task.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="Write the prompt here instead of stdout."
    ),
) -> None:
    """Start a task and return its prompt (the env's first yield)."""
    task_id, task_args, placement = _resolve(task, source, url, _parse_args(args))

    async def _run() -> dict[str, Any]:
        from hud.clients import connect

        # Start and disconnect without grading; an attached (persistent) env keeps
        # the session for a later `hud task grade` to resume.
        async with placement as runtime, connect(runtime) as client:
            return await client.start_task(task_id, task_args)

    _emit(asyncio.run(_run()), "prompt", out)


@task_app.command("grade")
def grade_command(
    task: str = typer.Argument(..., help="Task id or slug."),
    answer: str = typer.Option("", "--answer", help="Answer to grade."),
    answer_file: Path | None = typer.Option(  # noqa: B008
        None, "--answer-file", help="Read the answer from a file instead of --answer."
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Resolve the task from this source (.py/dir/JSON); spawn it unless --url is set.",
    ),
    args: str = typer.Option("{}", "--args", "-a", help="JSON object of task args."),
    url: str | None = typer.Option(
        None,
        "--url",
        "-u",
        help="Run against this served control channel; --source may still resolve the task.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="Write the full JSON result here (else print the reward)."
    ),
) -> None:
    """Grade an answer for a task and return its reward."""
    answer_text = answer_file.read_text(encoding="utf-8") if answer_file is not None else answer
    task_id, task_args, placement = _resolve(task, source, url, _parse_args(args))

    async def _run() -> dict[str, Any]:
        from hud.clients import connect
        from hud.clients.client import HudProtocolError

        async with placement as runtime, connect(runtime) as client:
            try:
                return await client.grade({"answer": answer_text})  # resume a prior start
            except HudProtocolError:
                # No held session: run the whole lifecycle here (start then grade).
                await client.start_task(task_id, task_args)
                return await client.grade({"answer": answer_text})

    _emit(asyncio.run(_run()), "score", out)


__all__ = ["task_app"]
