"""Contracts every ``hud`` command keeps, one table per contract over the command tree.

Every command in the tree is registered here: one that calls the platform with an
invocation that does, one that never does with an invocation that fails. A command
is also classified by what it changes, and a command with ``--quiet`` is a list
with a row in the list table. A new command fails the suite until it has its rows.

The contracts: a command that changes platform state asks before it does and sends
nothing on a dry run; a failure is one error document on stdout under ``--json``
and stays off stdout otherwise; a platform refusal maps to the same error kind
whichever command met it; an interrupt exits 130; every invocation posts one
analytics event naming its command; a list prints one id per line under
``--quiet``, from every page; a failure leaves the directory's link alone; a
secret given to a command never appears in what it prints.
"""

from __future__ import annotations

import json
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
import typer
from typer.core import TyperGroup

from hud.cli.__main__ import app
from tests.support import Reply, write_files
from tests.support.platform import BROWSER_PROJECT_ID, Import

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import Hud, HudEnv, Platform

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"
IMPORT_ID = "66666666-6666-4666-8666-666666666666"
SECRET = "s3cr3t-c0d3"
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


def a_platform_taskset(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_taskset("Demo", [{"name": "one", "env": "demo", "scenario": "solve"}])


def tasks(hud: Hud, platform: Platform) -> None:
    del platform
    (hud.cwd / "tasks.py").write_text(TASKS_PY)


def linked_taskset_with_a_change(hud: Hud, platform: Platform) -> None:
    tasks(hud, platform)
    assert hud("sync", "tasks", "demo", "tasks.py", "--yes").exit_code == 0
    (hud.cwd / "tasks.py").write_text(TASKS_PY.replace("solve(n=1)", "solve(n=1), solve(n=2)"))


def a_registered_environment(hud: Hud, platform: Platform) -> None:
    del hud
    platform.add_environment("example")


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
    taskset = platform.add_taskset("tb2")
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


def test_every_command_is_registered_and_classified() -> None:
    by_reach = [set(PLATFORM), set(OFFLINE)]
    by_change = [MUTATING, LOCAL_ONLY, READ_ONLY]

    for kinds in (by_reach, by_change):
        assert sorted(set(TREE) - set().union(*kinds)) == []
        assert sorted(set().union(*kinds) - set(TREE)) == []
        assert sum(len(kind) for kind in kinds) == len(TREE)


# ─── confirmation ───────────────────────────────────────────────────────


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


@pytest.mark.parametrize("name", sorted(MUTATING))
@pytest.mark.parametrize("flag", ["", "--dry-run"], ids=["unconfirmed", "dry-run"])
def test_a_mutating_command_changes_nothing_unless_confirmed(
    hud: Hud, platform: Platform, hud_env: HudEnv, name: str, flag: str
) -> None:
    command = PLATFORM[name]
    command.setup(hud, platform)
    before, sent = state(hud, hud_env), len(writes(platform))

    result = hud(*command.argv, *filter(None, [flag]), "--json")

    if flag:
        assert (result.exit_code, result.json["dry_run"]) == (0, True), result
    else:
        assert (result.exit_code, result.json["error"]) == (2, "usage"), result
        assert "--yes" in result.json["suggestion"]
    assert writes(platform)[sent:] == []
    assert state(hud, hud_env) == before


# ─── failures ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", EVERY_COMMAND)
def test_a_failure_is_one_error_document_under_json_and_stays_off_stdout_in_text(
    hud: Hud, platform: Platform, hud_env: HudEnv, name: str
) -> None:
    argv = failing(name, hud, platform, hud_env)

    as_json = hud(*argv, "--json")
    as_text = hud(*argv)

    error = as_json.json
    assert as_json.exit_code == (2 if error["error"] == "usage" else 1), as_json
    assert error["message"] not in as_json.stderr, as_json
    assert (as_text.exit_code, as_text.stdout) == (as_json.exit_code, ""), as_text
    assert error["message"] in as_text.stderr, as_text


ANSWERS = {
    "401": (Reply(status=401, json={"detail": "Refused"}), "permission_denied"),
    "403": (Reply(status=403, json={"detail": "Refused"}), "permission_denied"),
    "404": (Reply(status=404, json={"detail": "Refused"}), "not_found"),
    "409": (Reply(status=409, json={"detail": "Refused"}), "conflict"),
    "429": (Reply(status=429, json={"detail": "Refused"}), "rate_limited"),
    "500": (Reply(status=500, json={"detail": "Refused"}), "server_error"),
    "not-json": (Reply(body="<html>", content_type="text/html"), "failure"),
}


# Every command goes through one status mapping: each command is held to it with one
# status, and the mapping itself is read through one command.
REFUSALS = [(name, "403") for name in sorted(PLATFORM)] + [
    ("jobs get", answer) for answer in ANSWERS if answer != "403"
]


@pytest.mark.parametrize(("name", "answer"), REFUSALS)
def test_a_platform_refusal_is_the_same_error_kind_for_every_command(
    hud: Hud, platform: Platform, name: str, answer: str
) -> None:
    reply, error = ANSWERS[answer]
    PLATFORM[name].setup(hud, platform)
    answer_everything(platform, reply)

    result = hud(*invocation(name), "--json")

    assert (result.exit_code, result.json["error"]) == (1, error), result


def named(name: str) -> dict[str, Any]:
    command, _, subcommand = name.partition(" ")
    return {"command": command, "subcommand": subcommand or None}


@pytest.mark.parametrize("name", EVERY_COMMAND)
def test_every_invocation_posts_one_event_naming_its_command(
    hud: Hud, analytics: Platform, hud_env: HudEnv, name: str
) -> None:
    argv = failing(name, hud, analytics, hud_env)
    before = len(analytics.cli_events)

    result = hud(*argv)

    (event,) = analytics.cli_events[before:]
    assert {key: event[key] for key in ("command", "subcommand")} == named(name)
    assert event["exit_code"] == result.exit_code != 0
    assert event["error_class"] is not None


@pytest.mark.parametrize("name", sorted(set(PLATFORM) - {"deploy"}))
def test_an_interrupted_command_exits_130(
    hud: Hud, analytics: Platform, hud_env: HudEnv, name: str
) -> None:
    PLATFORM[name].setup(hud, analytics)
    answer_everything(analytics, Reply(json={}, delay=60))
    asked, posted = len(analytics.services.requests("api")), len(analytics.cli_events)
    process = hud.start(*invocation(name))

    deadline = time.monotonic() + 30
    while len(analytics.services.requests("api")) == asked:
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, f"hud {name} never reached the platform"
        time.sleep(0.05)
    process.send_signal(signal.SIGINT)
    process.communicate(timeout=30)

    assert process.returncode == 130
    (event,) = analytics.cli_events[posted:]
    assert event == {**event, **named(name), "exit_code": 130, "error_class": "KeyboardInterrupt"}


# ─── lists ──────────────────────────────────────────────────────────────

ENV_PY = """\
from hud import Environment

env = Environment("example")


@env.template(id="solve")
async def solve(target: str = "answer"):
    answer = yield f"Say {target}."
    yield 1.0 if answer == target else 0.0
"""
CHECKS = [
    {"key": "failure_analysis", "subject_type": "trace", "title": "Failure Analysis"},
    {"key": "reward_hacking", "subject_type": "trace", "title": "Reward Hacking"},
    {"key": "task_reward_exploit", "subject_type": "task", "title": "Task Reward Exploit"},
]


def python_tasks(hud: Hud, platform: Platform) -> list[str]:
    del platform
    (hud.cwd / "env.py").write_text(ENV_PY)
    (hud.cwd / "tasks.py").write_text(
        'from env import solve\ntasks = [solve(), solve(target="hello")]\n'
    )
    return ["solve", "solve-8da35614"]


def projects(hud: Hud, platform: Platform) -> list[str]:
    del hud
    platform.projects += [
        {"id": f"dddddddd-0000-4000-8000-{n:012d}", "name": f"p{n}"} for n in range(3)
    ]
    return [project["id"] for project in platform.projects]


def catalog(hud: Hud, platform: Platform) -> list[str]:
    del hud
    platform.add_model(name="owner/model + version", model_name="team/my-sonnet")
    platform.add_model(name="Claude Sonnet", model_name="claude-sonnet-4-6")
    return ["claude-sonnet-4-6", "team/my-sonnet"]


def checkpoints(hud: Hud, platform: Platform) -> list[str]:
    a_trainable_model(hud, platform)
    (model,) = platform.checkpoints.values()
    model.append({"id": "ckpt-2", "created_at": "2026-09-02T00:00:00Z"})
    return ["ckpt-1", "ckpt-2"]


def jobs(hud: Hud, platform: Platform) -> list[str]:
    del hud
    return [platform.add_job(name=f"job-{n}", status="completed") for n in range(3)]


def traces(hud: Hud, platform: Platform) -> list[str]:
    del hud
    ids = [f"00000000-0000-4000-a000-00000000000{n}" for n in range(2)]
    platform.add_job([{"id": trace, "status": "completed"} for trace in ids], id=JOB_ID)
    return ids


def checks(hud: Hud, platform: Platform) -> list[str]:
    del hud
    platform.qa_checks = CHECKS
    return [check["key"] for check in CHECKS]


def task_checks(hud: Hud, platform: Platform) -> list[str]:
    checks(hud, platform)
    return ["task_reward_exploit"]


def imports(hud: Hud, platform: Platform) -> list[str]:
    del hud
    taskset = platform.add_taskset("tb2")
    ids = [f"66666666-6666-4666-8666-00000000000{n}" for n in range(2)]
    for found in ids:
        platform.imports[found] = Import(found, taskset)
    return ids


LISTS: dict[str, tuple[list[str], Callable[[Hud, Platform], list[str]]]] = {
    "task list": (["task", "list", "--source", "tasks.py"], python_tasks),
    "project list": (["project", "list"], projects),
    "models list": (["models", "list"], catalog),
    "models checkpoints": (["models", "checkpoints", "team/my-sonnet"], checkpoints),
    "jobs": (["jobs"], jobs),
    "jobs list": (["jobs", "list"], jobs),
    "jobs get": (["jobs", "get", JOB_ID], traces),
    "qa": (["qa"], checks),
    "qa list": (["qa", "list", "--level", "task"], task_checks),
    "harbor imports": (["harbor", "imports", "--taskset", "tb2"], imports),
}


def test_every_command_with_quiet_is_a_list() -> None:
    quiet = {
        name
        for name, command in TREE.items()
        if any("--quiet" in getattr(param, "opts", []) for param in getattr(command, "params", []))
    }

    assert sorted(quiet) == sorted(LISTS)


@pytest.mark.parametrize("name", LISTS)
def test_a_list_prints_one_id_per_line_under_quiet_from_every_page(
    hud: Hud, platform: Platform, name: str
) -> None:
    """The platform pages its project and model catalogs; one item a page makes a list
    of several read every page."""
    argv, seed = LISTS[name]
    expected = seed(hud, platform)
    platform.page_size = 1

    result = hud(*argv, "--quiet")

    assert (result.exit_code, result.stdout.split("\n")[:-1]) == (0, expected), result


# ─── the directory link ─────────────────────────────────────────────────


def refused(method: str, path: str, status: int) -> Setup:
    def setup(hud: Hud, platform: Platform) -> None:
        del hud
        platform.services.route("api", method, path, Reply(status=status, json={"detail": "No"}))

    return setup


LINK_WRITERS: dict[str, tuple[list[str], list[Setup], str]] = {
    "deploy": (
        ["deploy", "--no-env", "--yes"],
        [environment, refused("POST", "/v2/builds/trigger", 409)],
        "conflict",
    ),
    "sync": (
        ["sync", "--yes"],
        [linked_taskset_with_a_change, refused("POST", "/v2/tasks/upload", 400)],
        "failure",
    ),
    "sync tasks": (
        ["sync", "tasks", "demo", "tasks.py", "--yes"],
        [tasks, refused("POST", "/v2/tasks/upload", 400)],
        "failure",
    ),
    "sync env": (["sync", "env", "bbbbbbbb-0000-4000-8000-999999999999"], [], "not_found"),
    "project create": (
        ["project", "create", "UI Evals", "--yes"],
        [refused("POST", "/v2/projects", 403)],
        "permission_denied",
    ),
    "project use": (["project", "use", "locked-down"], [], "permission_denied"),
    "harbor import": (
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2", "--link", "--yes"],
        [harbor_folders, refused("POST", "/v2/tasksets", 403)],
        "permission_denied",
    ),
}


@pytest.mark.parametrize("name", LINK_WRITERS)
def test_a_command_that_fails_leaves_the_directory_link_unchanged(
    hud: Hud, platform: Platform, name: str
) -> None:
    argv, setups, error = LINK_WRITERS[name]
    link = hud.cwd / ".hud" / "config.json"
    write_files(hud.cwd, {".hud/config.json": json.dumps({"project_id": BROWSER_PROJECT_ID})})
    for setup in setups:
        setup(hud, platform)
    before = link.read_bytes()

    result = hud(*argv, "--json")

    assert (result.exit_code, result.json["error"]) == (1, error), result
    assert link.read_bytes() == before


# ─── secrets ────────────────────────────────────────────────────────────


def secrets_beside_an_environment(hud: Hud, platform: Platform) -> None:
    environment(hud, platform)
    write_files(hud.cwd, {".env": f"SECRET={SECRET}\n", "token.txt": SECRET})


SECRETS: dict[str, tuple[list[str], list[Setup], dict[str, str], str]] = {
    "set": (["set", f"HUD_API_KEY={SECRET}"], [], {}, SECRET),
    "deploy": (
        [
            *("deploy", "--yes"),
            *("--secret", "id=ENV_TOKEN,env=ENV_TOKEN"),
            *("--secret", "id=FILE_TOKEN,src=token.txt"),
        ],
        [secrets_beside_an_environment],
        {"ENV_TOKEN": SECRET},
        SECRET,
    ),
    "eval": (
        [
            *("eval", "tasks.py", "openai_compatible", "-m", "custom"),
            *("-c", f"api_key={SECRET}", "-c", "base_url=http://127.0.0.1:9", "--dry-run"),
        ],
        [tasks],
        {},
        SECRET,
    ),
    "harbor import": (
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2", "--yes"],
        [harbor_folders],
        {},
        "signature",
    ),
    "harbor import, upload refused": (
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2", "--yes"],
        [harbor_folders, refused("PUT", "/uploads/{id}", 403)],
        {},
        "signature",
    ),
}


@pytest.mark.parametrize("name", SECRETS)
@pytest.mark.parametrize("mode", [[], ["--json"]], ids=["text", "json"])
def test_a_secret_given_to_a_command_never_appears_in_its_output(
    hud: Hud, platform: Platform, name: str, mode: list[str]
) -> None:
    argv, setups, env, secret = SECRETS[name]
    for setup in setups:
        setup(hud, platform)

    result = hud(*argv, *mode, env=env)

    assert secret not in result.stdout + result.stderr, result
