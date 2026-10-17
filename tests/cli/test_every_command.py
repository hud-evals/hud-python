"""Contracts every ``hud`` command keeps, one table over the command tree.

A command that changes platform state asks before it does, refuses in a
non-interactive terminal unless ``--yes`` is passed, and with ``--dry-run``
sends and writes nothing. Every command in the tree is classified here, so a
new one fails until it says which kind it is.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
import typer
from typer.core import TyperGroup

from hud.cli.__main__ import app
from tests.support import write_files
from tests.support.platform import Import

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import Hud, HudEnv, Platform

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"
IMPORT_ID = "66666666-6666-4666-8666-666666666666"
TASKS_PY = """\
from hud import Environment

env = Environment("example")


@env.template(id="solve")
async def solve(n: int = 0):
    yield f"solve {n}"
    yield 1.0


tasks = [solve(n=1)]
"""

Setup = Callable[["Hud", "Platform"], None]


def nothing(hud: Hud, platform: Platform) -> None:
    del hud, platform


def environment(hud: Hud, platform: Platform) -> None:
    del platform
    write_files(
        hud.cwd,
        {
            "env.py": 'from hud import Environment\nenv = Environment("example")\n',
            "Dockerfile.hud": "FROM python:3.12\n",
        },
    )


def tasks(hud: Hud, platform: Platform) -> None:
    del platform
    (hud.cwd / "tasks.py").write_text(TASKS_PY)


def linked_taskset_with_a_change(hud: Hud, platform: Platform) -> None:
    tasks(hud, platform)
    assert hud("sync", "tasks", "demo", "tasks.py", "--yes").exit_code == 0
    (hud.cwd / "tasks.py").write_text(TASKS_PY.replace("solve(n=1)", "solve(n=1), solve(n=2)"))


def harbor_folders(hud: Hud, platform: Platform) -> None:
    del platform
    write_files(
        hud.cwd,
        {
            "tasks/hello/task.toml": "version = '1.0'\n",
            "tasks/hello/tests/test.sh": "exit 0\n",
            "run-1/config.json": "{}",
            "run-1/result.json": "{}",
        },
    )


def a_running_import(hud: Hud, platform: Platform) -> None:
    del hud
    platform.imports[IMPORT_ID] = Import(IMPORT_ID, "tb2", statuses=["running"])


def a_qa_check(hud: Hud, platform: Platform) -> None:
    del hud
    platform.qa_checks = [{"key": "failure_analysis", "subject_type": "trace", "title": "FA"}]


def a_base_model(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_model(
        name="Claude Sonnet",
        model_name="claude-sonnet-4-6",
        sdk_agent_type="claude",
        provider={"name": "anthropic"},
    )


def a_trainable_model(hud: Hud, platform: Platform) -> None:
    del hud
    model_id = platform.add_model(
        name="Mine",
        model_name="team/my-sonnet",
        sdk_agent_type="claude",
        is_trainable=True,
        provider={"name": "anthropic"},
    )
    platform.checkpoints[model_id] = [{"id": "ckpt-1", "created_at": "2026-09-01T00:00:00Z"}]


MUTATING: dict[str, tuple[list[str], Setup]] = {
    "deploy": (["deploy", "--no-env"], environment),
    "eval": (["eval", "tasks.py", "openai_compatible", "-m", "scripted", "--all"], tasks),
    "project create": (["project", "create", "UI Evals"], nothing),
    "sync": (["sync"], linked_taskset_with_a_change),
    "sync tasks": (["sync", "tasks", "demo", "tasks.py"], tasks),
    "harbor import": (
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2"],
        harbor_folders,
    ),
    "harbor cancel": (["harbor", "cancel", IMPORT_ID], a_running_import),
    "qa run": (["qa", "run", "failure_analysis", TRACE_ID], a_qa_check),
    "jobs cancel": (["jobs", "cancel", JOB_ID], nothing),
    "cancel": (["cancel", JOB_ID, "--trace-id", TRACE_ID], nothing),
    "models fork": (["models", "fork", "claude-sonnet-4-6", "--name", "mine"], a_base_model),
    "models head": (["models", "head", "team/my-sonnet", "--set", "ckpt-1"], a_trainable_model),
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


def commands(command: object, path: tuple[str, ...] = ()) -> set[str]:
    """Every invocable command path; a group that runs on its own counts as one."""
    found = (
        {" ".join(path)} if path and getattr(command, "invoke_without_command", False) else set()
    )
    if isinstance(command, TyperGroup):
        for name, sub in command.commands.items():
            found |= commands(sub, (*path, name))
    elif path:
        found.add(" ".join(path))
    return found


def test_every_command_is_classified_by_what_it_changes() -> None:
    tree = commands(typer.main.get_command(app))

    classified = [set(MUTATING), LOCAL_ONLY, READ_ONLY]
    assert sorted(tree - set().union(*classified)) == []
    assert sorted(set().union(*classified) - tree) == []
    assert sum(len(kind) for kind in classified) == len(set().union(*classified))


def state(hud: Hud, hud_env: HudEnv) -> dict[Path, bytes]:
    roots = (hud.cwd, hud_env.home / ".hud")
    return {
        path: path.read_bytes()
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def writes(platform: Platform) -> list[str]:
    return [
        f"{request.method} {request.path}"
        for request in platform.services.requests()
        if request.method != "GET" and request.service != "telemetry"
    ]


@pytest.mark.parametrize("name", MUTATING)
@pytest.mark.parametrize("flag", ["", "--dry-run"], ids=["unconfirmed", "dry-run"])
def test_a_mutating_command_changes_nothing_unless_confirmed(
    hud: Hud, platform: Platform, hud_env: HudEnv, name: str, flag: str
) -> None:
    argv, setup = MUTATING[name]
    setup(hud, platform)
    before, sent = state(hud, hud_env), len(writes(platform))

    result = hud(*argv, *filter(None, [flag]), "--json")

    if flag:
        assert (result.exit_code, result.json["dry_run"]) == (0, True), result
    else:
        assert (result.exit_code, result.json["error"]) == (2, "usage"), result
        assert "--yes" in result.json["suggestion"]
    assert writes(platform)[sent:] == []
    assert state(hud, hud_env) == before
