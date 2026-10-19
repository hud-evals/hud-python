"""The ``hud`` command tree as the rows of the CLI contracts.

Every invocable command is registered here twice over: by how it reaches the
platform (``PLATFORM`` with an invocation that calls it, ``OFFLINE`` with one that
fails without calling anything) and by what it changes (``MUTATING``,
``LOCAL_ONLY``, ``READ_ONLY``). ``test_shell`` holds the registry to the tree, so
a new command fails the suite until it has its rows.

A setup takes the ``hud`` runner and the platform and puts in place what an
invocation needs: files in the working directory, platform records, routes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import typer
from typer.core import TyperGroup

from hud.cli.__main__ import app
from tests.support import Reply, write_files
from tests.support.platform import Import

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import Hud, HudEnv, Platform

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"
IMPORT_ID = "66666666-6666-4666-8666-666666666666"
TB2_ID = "eeeeeeee-0000-4000-8000-999999999999"
ENV_ID = "bbbbbbbb-0000-4000-8000-000000000042"
BASE_MODEL_ID = "cccccccc-0000-4000-8000-000000000101"
MODEL_ID = "cccccccc-0000-4000-8000-000000000102"
TASKS_PY = """\
from hud import Environment

env = Environment("example")


@env.template(id="solve")
async def solve(n: int = 0):
    yield f"solve {n}"
    yield 1.0


tasks = [solve(n=1)]
"""

Setup = Callable[["Hud", "Platform"], object]


def nothing(hud: Hud, platform: Platform) -> None:
    del hud, platform


def files(contents: dict[str, str]) -> Setup:
    """Write ``contents`` under the working directory; ``->target`` makes a symlink."""

    def setup(hud: Hud, platform: Platform) -> None:
        del platform
        for name, content in contents.items():
            if content.startswith("->"):
                path = hud.cwd / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.symlink_to(hud.cwd / content.removeprefix("->"))
            else:
                write_files(hud.cwd, {name: content})

    return setup


def link(stored: object) -> Setup:
    """A directory link (``.hud/config.json``) holding ``stored``, written as JSON."""
    return files({".hud/config.json": stored if isinstance(stored, str) else json.dumps(stored)})


def refused(method: str, path: str, status: int) -> Setup:
    def setup(hud: Hud, platform: Platform) -> None:
        del hud
        platform.services.route("api", method, path, Reply(status=status, json={"detail": "No"}))

    return setup


def together(*setups: Setup) -> Setup:
    def setup(hud: Hud, platform: Platform) -> None:
        for each in setups:
            each(hud, platform)

    return setup


environment = files(
    {
        "env.py": 'from hud import Environment\nenv = Environment("example")\n',
        "Dockerfile.hud": "FROM python:3.12\n",
    }
)
tasks = files({"tasks.py": TASKS_PY})
harbor_folders = files(
    {
        "tasks/hello/task.toml": "version = '1.0'\n",
        "tasks/hello/tests/test.sh": "exit 0\n",
        "run-1/config.json": "{}",
        "run-1/result.json": "{}",
    }
)


def a_platform_taskset(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_taskset("Demo", [{"name": "one", "env": "demo", "scenario": "solve"}])


def linked_taskset_with_a_change(hud: Hud, platform: Platform) -> None:
    tasks(hud, platform)
    assert hud("sync", "tasks", "demo", "tasks.py", "--yes").exit_code == 0
    (hud.cwd / "tasks.py").write_text(TASKS_PY.replace("solve(n=1)", "solve(n=1), solve(n=2)"))


def a_registered_environment(hud: Hud, platform: Platform) -> str:
    del hud
    return platform.add_environment("example", id=ENV_ID)


def a_running_import(hud: Hud, platform: Platform) -> None:
    del hud
    taskset = platform.add_taskset("tb2", id=TB2_ID)
    platform.imports[IMPORT_ID] = Import(
        IMPORT_ID, taskset, statuses=["running"], report={"schema_version": "v9"}
    )


def a_qa_check(hud: Hud, platform: Platform) -> None:
    del hud
    platform.qa_checks = [{"key": "failure_analysis", "subject_type": "trace", "title": "FA"}]


def a_job(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_job([{"id": TRACE_ID, "status": "completed"}], id=JOB_ID)


def a_trace(hud: Hud, platform: Platform) -> None:
    del hud
    platform.trace(TRACE_ID).events = [{"kind": "agent_message", "text": "done"}]


def a_base_model(hud: Hud, platform: Platform) -> str:
    del hud
    return platform.add_model(
        id=BASE_MODEL_ID,
        name="Claude Sonnet",
        model_name="claude-sonnet-4-6",
        sdk_agent_type="claude",
        provider={"name": "anthropic"},
    )


def a_trainable_model(hud: Hud, platform: Platform) -> str:
    del hud
    model_id = platform.add_model(
        id=MODEL_ID,
        name="owner/model + version",
        model_name="team/my-sonnet",
        sdk_agent_type="claude",
        is_trainable=True,
        provider={"name": "anthropic"},
    )
    platform.checkpoints[model_id] = [
        {"id": "ckpt-1", "created_at": "2026-09-01T00:00:00Z"},
        {"id": "ckpt-2", "name": "step-1", "created_at": "2026-09-02T00:00:00Z", "is_active": True},
    ]
    return model_id


@dataclass(frozen=True)
class Command:
    """An invocation that reaches the platform, and the state it needs first."""

    argv: list[str]
    setup: Setup = nothing


PLATFORM: dict[str, Command] = {
    "deploy": Command(["deploy", "--no-env"], environment),
    "eval": Command(["eval", "Demo", "openai_compatible", "-m", "scripted"], a_platform_taskset),
    "cancel": Command(["cancel", JOB_ID, "--trace-id", TRACE_ID]),
    "project": Command(["project"]),
    "project list": Command(["project", "list"]),
    "project create": Command(["project", "create", "UI Evals"]),
    "project use": Command(["project", "use", "browser-evals"]),
    "sync": Command(["sync"], linked_taskset_with_a_change),
    "sync tasks": Command(["sync", "tasks", "demo", "tasks.py"], tasks),
    "sync env": Command(["sync", "env", "example"], a_registered_environment),
    "harbor import": Command(
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2"], harbor_folders
    ),
    "harbor imports": Command(["harbor", "imports", "--taskset", "tb2"], a_running_import),
    "harbor report": Command(["harbor", "report", IMPORT_ID], a_running_import),
    "harbor cancel": Command(["harbor", "cancel", IMPORT_ID], a_running_import),
    "qa": Command(["qa"], a_qa_check),
    "qa list": Command(["qa", "list"], a_qa_check),
    "qa run": Command(["qa", "run", "failure_analysis", TRACE_ID], a_qa_check),
    "qa results": Command(["qa", "results", TRACE_ID]),
    "jobs": Command(["jobs"]),
    "jobs list": Command(["jobs", "list"]),
    "jobs get": Command(["jobs", "get", JOB_ID], a_job),
    "jobs cancel": Command(["jobs", "cancel", JOB_ID]),
    "trace get": Command(["trace", "get", TRACE_ID], a_trace),
    "models list": Command(["models", "list"], a_base_model),
    "models fork": Command(["models", "fork", "claude-sonnet-4-6", "--name", "mine"], a_base_model),
    "models checkpoints": Command(["models", "checkpoints", "team/my-sonnet"], a_trainable_model),
    "models head": Command(
        ["models", "head", "team/my-sonnet", "--set", "ckpt-1"], a_trainable_model
    ),
}
OFFLINE: dict[str, list[str]] = {
    "init": ["init", "thing", "--preset", "no-such-preset"],
    "serve": ["serve", "missing.py"],
    "set": ["set", "NOT_A_PAIR"],
    "version": ["version", "--no-such-option"],
    "task list": ["task", "list", "--source", "missing.json"],
    "task start": ["task", "start", "solve", "--source", "missing.json"],
    "task grade": ["task", "grade", "solve", "--source", "missing.json", "--answer", "x"],
}

MUTATING = {
    "deploy",
    "eval",
    "project create",
    "sync",
    "sync tasks",
    "harbor import",
    "harbor cancel",
    "qa run",
    "jobs cancel",
    "cancel",
    "models fork",
    "models head",
}
LOCAL_ONLY = {"init", "project use", "sync env", "set", "serve", "task start", "task grade"}
READ_ONLY = {
    "version",
    "task list",
    "project",
    "project list",
    "harbor imports",
    "harbor report",
    "qa",
    "qa list",
    "qa results",
    "jobs",
    "jobs list",
    "jobs get",
    "trace get",
    "models list",
    "models checkpoints",
}


def tree(command: object, path: tuple[str, ...] = ()) -> dict[str, object]:
    """Every invocable command by path; a group that runs on its own counts as one."""
    group = isinstance(command, TyperGroup)
    runs = path and (not group or getattr(command, "invoke_without_command", False))
    found = {" ".join(path): command} if runs else {}
    if isinstance(command, TyperGroup):
        for name, sub in command.commands.items():
            found |= tree(sub, (*path, name))
    return found


TREE = tree(typer.main.get_command(app))
EVERY_COMMAND = sorted(TREE)


def invocation(name: str) -> list[str]:
    """The registered platform invocation, confirmed when the command asks first."""
    return [*PLATFORM[name].argv, *(["--yes"] if name in MUTATING else [])]


def failing(name: str, hud: Hud, platform: Platform, hud_env: HudEnv) -> list[str]:
    """An invocation of ``name`` that fails: offline, its registered one; else signed out."""
    if name in OFFLINE:
        return OFFLINE[name]
    PLATFORM[name].setup(hud, platform)
    hud_env.set(HUD_API_KEY=None)
    return invocation(name)


def answer_everything(platform: Platform, reply: Reply) -> None:
    """Every platform API route answers ``reply``."""
    for method in ("GET", "POST", "PUT", "DELETE"):
        platform.services.route("api", method, "/{rest:path}", reply)


def state(hud: Hud, hud_env: HudEnv) -> dict[Path, bytes | str]:
    """Every file under the working directory and the user's ``~/.hud``; a symlink as its
    target."""
    roots = (hud.cwd, hud_env.home / ".hud")
    return {
        path: str(path.readlink()) if path.is_symlink() else path.read_bytes()
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink()) and "__pycache__" not in path.parts
    }


def writes(platform: Platform) -> list[str]:
    """The platform requests that could change something: every API call but a read."""
    return [
        f"{request.method} {request.path}"
        for request in platform.services.requests()
        if request.method != "GET" and request.service != "telemetry"
    ]
