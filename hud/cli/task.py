"""``hud task`` — start a task (get its prompt) or grade an answer.

Direct by default: introspects the local env source (the same ``.py``/dir/JSON the
``hud eval`` flow collects ``Task``s from) and runs the task **in-process** — no
served daemon, no port, no protocol on the wire. Pass ``--url`` to attach to an
already-served control channel instead.

    hud task list                          # what tasks this source/image exposes
    hud task start fix_config              # -> the task's prompt (stdout)
    hud task grade fix_config --answer "…" # -> the reward (stdout); --out for JSON
"""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path  # noqa: TC003 - Typer resolves the `Path` option annotations at runtime
from typing import Any
from urllib.parse import urlsplit

import typer

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()

task_app = typer.Typer(
    help="Start a task or grade an answer (attaches to a running env, or runs from source).",
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
    (e.g. ``hud dev``, or a built image whose CMD serves on :8765), else ``None``."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return f"tcp://127.0.0.1:{port}"
    except OSError:
        return None


def _resolve_task(task: str, source: str | None, url: str | None, args: dict[str, Any]) -> Any:
    """Build a runnable ``Task`` for ``task``, choosing a substrate in priority order:

    1. ``--url`` — attach to that control channel;
    2. no ``--source`` and a local env already serving on :8765 — attach to it
       (e.g. inside a built image, or alongside ``hud dev``);
    3. otherwise — introspect local source, matching by task id or slug.

    ``--args`` (when given) mints a fresh task on the chosen env so any
    parameterization is runnable.
    """
    from hud.eval import RemoteSandbox, Task

    attach = url
    if attach is None and source is None:
        attach = _local_env_url()
    if attach is not None:
        parts = urlsplit(attach if "://" in attach else f"tcp://{attach}")
        endpoint = f"tcp://{parts.hostname or '127.0.0.1'}:{parts.port or 8765}"
        return Task(env=RemoteSandbox(endpoint), id=task, args=args)

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
    # Override args onto the same env so an explicit parameterization is runnable.
    return Task(env=selected.env, id=selected.id, args=args) if args else selected


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
        None, "--source", "-s", help="Run from this env source (.py/dir/JSON) instead of attaching."
    ),
    args: str = typer.Option("{}", "--args", "-a", help="JSON object of task args."),
    url: str | None = typer.Option(
        None, "--url", "-u", help="Attach to a served control channel instead of loading source."
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="Write the prompt here instead of stdout."
    ),
) -> None:
    """Start a task and return its prompt (the env's first yield)."""
    runnable = _resolve_task(task, source, url, _parse_args(args))

    async def _run() -> dict[str, Any]:
        from hud.eval.launch import launch

        # Start and disconnect without grading; a persistent env keeps the session
        # for a later `hud task grade` to resume.
        async with launch(runnable.env) as client:
            return await client.start_task(runnable.id, runnable.args)

    _emit(asyncio.run(_run()), "prompt", out)


@task_app.command("grade")
def grade_command(
    task: str = typer.Argument(..., help="Task id or slug."),
    answer: str = typer.Option("", "--answer", help="Answer to grade."),
    answer_file: Path | None = typer.Option(  # noqa: B008
        None, "--answer-file", help="Read the answer from a file instead of --answer."
    ),
    source: str | None = typer.Option(
        None, "--source", "-s", help="Run from this env source (.py/dir/JSON) instead of attaching."
    ),
    args: str = typer.Option("{}", "--args", "-a", help="JSON object of task args."),
    url: str | None = typer.Option(
        None, "--url", "-u", help="Attach to a served control channel instead of loading source."
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="Write the full JSON result here (else print the reward)."
    ),
) -> None:
    """Grade an answer for a task and return its reward."""
    answer_text = answer_file.read_text(encoding="utf-8") if answer_file is not None else answer
    runnable = _resolve_task(task, source, url, _parse_args(args))

    async def _run() -> dict[str, Any]:
        from hud.client.client import HudProtocolError
        from hud.eval.launch import launch

        async with launch(runnable.env) as client:
            try:
                return await client.grade({"answer": answer_text})  # resume a prior start
            except HudProtocolError:
                # No held session: run the whole lifecycle here (start then grade).
                await client.start_task(runnable.id, runnable.args)
                return await client.grade({"answer": answer_text})

    _emit(asyncio.run(_run()), "score", out)


__all__ = ["task_app"]
