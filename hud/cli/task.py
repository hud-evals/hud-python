"""``hud task`` — start a task (get its prompt) or grade an answer."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from hud.cli import (
    CLI,
    CliError,
)
from hud.eval.taskset import Taskset
from hud.runtime import Runtime, SubprocessRuntime
from hud.utils.process import run_until_terminated

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

# How long a session `hud task start` leaves on a served env stays open for `hud task grade`.
_SESSION_LEASE_SECONDS = 24 * 60 * 60


def _sessions_file() -> Path:
    # The sessions `hud task start` left open, by env url and task, for `hud task grade`.
    return Path.home() / ".hud" / "task-sessions.json"


def _load_sessions() -> dict[str, str]:
    path = _sessions_file()
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def _save_sessions(sessions: dict[str, str]) -> None:
    path = _sessions_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sessions, indent=2), "utf-8")


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
    # A served env keeps the session open for `hud task grade`; a spawned one exits with us.
    lease = _SESSION_LEASE_SECONDS if url is not None else None

    async def _run() -> dict[str, Any]:
        async with placement as runtime, runtime.connect(lease=lease) as session:
            started = await session.start_task(task_id, task_args)
            return {**started, "session_id": session.session_id}

    result = run_until_terminated(_run())
    if url is not None:
        sessions = _load_sessions()
        sessions[f"{url} {task_id}"] = result["session_id"]
        _save_sessions(sessions)
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
    session: str | None = typer.Option(
        None,
        "--session",
        help="Grade the task in this session on --url (default: the one `hud task start` "
        "left there).",
    ),
) -> Any:
    """Grade an answer for a task and print its reward.

    [not dim]Examples:
        hud task grade fix_bug --answer "done"
        hud task grade fix_bug --answer-file - --json
        hud task grade fix_bug --answer-file answer.txt[/not dim]
    """
    answer_text = CLI.read_text(answer_file) if answer_file is not None else answer
    if session is not None and url is None:
        raise CliError(error="usage", message="--session needs the --url of the env serving it")
    task_id, task_args, placement = _resolve(task, source, url, args)
    if url is not None and session is None:
        # Attaching closes the session when this command ends, so it is forgotten either way.
        sessions = _load_sessions()
        session = sessions.pop(f"{url} {task_id}", None)
        _save_sessions(sessions)

    async def _run() -> dict[str, Any]:
        if session is not None:
            async with placement as runtime, runtime.connect(session_id=session) as attached:
                return await attached.grade({"answer": answer_text})
        # No session to grade in: run the whole lifecycle here.
        async with placement as runtime, runtime.connect() as fresh:
            await fresh.start_task(task_id, task_args)
            return await fresh.grade({"answer": answer_text})

    result = run_until_terminated(_run())
    typer.echo(json.dumps(result.get("score", result), default=str))
    return result
