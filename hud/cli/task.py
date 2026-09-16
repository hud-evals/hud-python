"""``hud task`` — start a task (get its prompt) or grade an answer.

The task source resolves an authored slug to its template id and bound args.
Without ``--url`` that source is also spawned locally; with ``--url`` the task
runs against the already-served control channel instead.

    hud task list                          # what tasks this source exposes
    hud task start fix_config              # -> the task's prompt (stdout)
    hud task grade fix_config --answer "…" # -> the reward (stdout); --json for the frame
"""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    """``(task_id, args, placement)`` for ``start`` and ``grade``.

    ``--url`` alone runs ``task`` as a raw template id with ``--args``. A source
    (``--source``, default ``.``) resolves an authored slug/id/index to its template
    id and bound args, and is spawned unless ``--url`` names a served env.
    """
    if url is not None and source is None:
        return task, args or {}, nullcontext(Runtime(url))

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
    if url is not None:
        placement: AbstractAsyncContextManager[Runtime] = nullcontext(Runtime(url))
    elif selected._env is not None:
        placement = SubprocessRuntime(selected._env)(selected)
    else:
        # A data row (JSON/JSONL) names its env; the env source lives beside the file.
        path = Path(source or ".").resolve()
        placement = SubprocessRuntime(path if path.is_dir() else path.parent)(selected)
    return selected.id, selected.args if args is None else args, placement


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
        help="Run against this served control channel (tcp://host:port); --source may still "
        "resolve the task.",
    ),
) -> Any:
    """Start a task and print its prompt (the env's first yield).

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

    result = asyncio.run(_run())
    prompt = result.get("prompt", result)
    typer.echo(prompt if isinstance(prompt, str) else json.dumps(prompt, default=str))
    return result


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
        help="Run against this served control channel (tcp://host:port); --source may still "
        "resolve the task.",
    ),
) -> Any:
    """Grade an answer for a task and print its reward.

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

    result = asyncio.run(_run())
    typer.echo(json.dumps(result.get("score", result), default=str))
    return result
