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
from contextlib import nullcontext
from pathlib import Path  # noqa: TC003 - Typer resolves command annotations at runtime.
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import typer

from hud.cli import (
    CLI,
    CliError,
)
from hud.clients import HudProtocolError, connect
from hud.eval import Taskset
from hud.eval.runtime import Runtime, SubprocessRuntime

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

task_app = CLI(
    help="Start a task or grade an answer (attaches to a running env, or spawns from source).",
    rich_markup_mode="rich",
)


def _resolve(
    task: str, source: str | None, url: str | None, args: dict[str, Any] | None
) -> tuple[str, dict[str, Any], AbstractAsyncContextManager[Runtime]]:
    """Resolve ``(task_id, args, placement)`` for ``start`` and ``grade``.

    ``--source`` resolves an authored task id/slug and its bound args. ``--url``
    selects an existing substrate; otherwise an explicit source is spawned. With
    neither option, a local env on :8765 is used when present, or ``.`` is resolved
    and spawned. ``--args`` overrides authored args when supplied.
    """
    attach = url
    if attach is None and source is None:
        # An env already serving locally (hud serve, or a built image's CMD).
        try:
            with socket.create_connection(("127.0.0.1", 8765), timeout=0.25):
                attach = "tcp://127.0.0.1:8765"
        except OSError:
            attach = None
    endpoint: Runtime | None = None
    if attach is not None:
        parts = urlsplit(attach if "://" in attach else f"tcp://{attach}")
        if parts.scheme != "tcp":
            raise CliError(error="usage", message="Task control channels require a tcp:// URL")
        host = parts.hostname or "127.0.0.1"
        endpoint = Runtime(f"tcp://{f'[{host}]' if ':' in host else host}:{parts.port or 8765}")

    if endpoint is not None and source is None:
        return task, args or {}, nullcontext(endpoint)

    taskset = Taskset.from_file(source or ".")
    if not taskset:
        raise CliError(
            error="not_found",
            message=f"No tasks found in {source or '.'}",
            input={"source": source or "."},
        )
    matches = [
        candidate
        for index, (slug, candidate) in enumerate(taskset.items())
        if task in (slug, candidate.id, str(index))
    ]
    if not matches:
        available = ", ".join(sorted({t.id for t in taskset}))
        raise CliError(
            error="not_found",
            message=f"No task matching {task!r} (available: {available})",
            input={"task": task, "source": source or "."},
            suggestion="Run 'hud task list' to see available slugs.",
        )
    if len(matches) > 1:
        raise CliError(
            error="usage",
            message=f"Ambiguous task {task!r}; use a unique slug shown by hud task list.",
        )
    selected = matches[0]
    if endpoint is not None:
        placement: AbstractAsyncContextManager[Runtime] = nullcontext(endpoint)
    elif selected._env is None:
        raise CliError(
            error="usage",
            message="These rows have no bound Environment.",
            input={"source": source or "."},
            suggestion="Pass --source to a tasks.py / env.py that binds an Environment, "
            "or --url to attach to a served env.",
        )
    else:
        placement = SubprocessRuntime(selected._env)(selected)
    return selected.id, selected.args if args is None else args, placement


def _emit(result: dict[str, Any], headline: str, out: Path | None) -> dict[str, Any] | None:
    """Write the full frame to ``--out``, otherwise the headline value to stdout."""
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return None
    value = result.get(headline, result)
    typer.echo(value if isinstance(value, str) else json.dumps(value, default=str))
    return result


@task_app.command("list")
def list_command(
    source: str = typer.Option(".", "--source", "-s", help="Env source (.py/dir/JSON)."),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print one identifier per line, with no headers (for piping)."
    ),
) -> Any:
    """List the tasks (slug + task id + args) exposed by a source.

    [not dim]Examples:
        hud task list
        hud task list --json
        hud task list --quiet[/not dim]
    """
    items = [
        {"slug": slug, "id": task.id, "args": task.args}
        for slug, task in Taskset.from_file(source).items()
    ]
    for item in items:
        if quiet:
            typer.echo(item["slug"])
        else:
            args = f" {json.dumps(item['args'])}" if item["args"] else ""
            typer.echo(f"{item['slug']}\t{item['id']}{args}")
    return items


@task_app.command("start")
def start_command(
    task: str = typer.Argument(..., help="Task id or slug."),
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Resolve the task from this source (.py/dir/JSON); spawn it unless --url is set.",
    ),
    args: dict[str, Any] | None = typer.Option(  # noqa: B008
        None,
        "--args",
        "-a",
        help="JSON object of task args.",
        parser=lambda value: CLI.json_object(value, option="--args"),
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        "-u",
        help="Run against this served control channel; --source may still resolve the task.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="Write the prompt here instead of stdout."
    ),
) -> Any:
    """Start a task and return its prompt (the env's first yield).

    [not dim]Examples:
        hud task start fix_bug
        hud task start fix_bug --json
        hud task start fix_bug --source . --args '{}'[/not dim]
    """
    task_id, task_args, placement = _resolve(task, source, url, args)

    async def _run() -> dict[str, Any]:
        # Start and disconnect without grading; an attached (persistent) env keeps
        # the session for a later `hud task grade` to resume.
        async with placement as runtime, connect(runtime) as client:
            return await client.start_task(task_id, task_args)

    return _emit(asyncio.run(_run()), "prompt", out)


@task_app.command("grade")
def grade_command(
    task: str = typer.Argument(..., help="Task id or slug."),
    answer: str = typer.Option("", "--answer", help="Answer to grade."),
    answer_file: str | None = typer.Option(
        None,
        "--answer-file",
        help="Read the answer from a file instead of --answer. Pass - to read stdin.",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        "-s",
        help="Resolve the task from this source (.py/dir/JSON); spawn it unless --url is set.",
    ),
    args: dict[str, Any] | None = typer.Option(  # noqa: B008
        None,
        "--args",
        "-a",
        help="JSON object of task args.",
        parser=lambda value: CLI.json_object(value, option="--args"),
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        "-u",
        help="Run against this served control channel; --source may still resolve the task.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None, "--out", "-o", help="Write the full JSON result here (else print the reward)."
    ),
) -> Any:
    """Grade an answer for a task and return its reward.

    [not dim]Examples:
        hud task grade fix_bug --answer "done"
        hud task grade fix_bug --answer-file - --json
        hud task grade fix_bug --answer-file answer.txt[/not dim]
    """
    answer_text = CLI.read_text(answer_file) if answer_file is not None else answer
    task_id, task_args, placement = _resolve(task, source, url, args)

    async def _run() -> dict[str, Any]:
        async with placement as runtime, connect(runtime) as client:
            try:
                return await client.grade({"answer": answer_text})  # resume a prior start
            except HudProtocolError as exc:
                if exc.code != -32600 or exc.message != "no task in progress":
                    raise
                # No held session: run the whole lifecycle here (start then grade).
                await client.start_task(task_id, task_args)
                return await client.grade({"answer": answer_text})

    return _emit(asyncio.run(_run()), "score", out)
