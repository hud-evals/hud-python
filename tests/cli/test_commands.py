"""What every ``hud`` command does alike: the command tree itself, help and unknown
commands, the upgrade notice, anonymous analytics, an interrupt, asking before it changes
anything, the error document a failure prints, the error kind a platform refusal maps to,
one id per line under ``--quiet``, and never echoing a secret it was given.

Each command in the tree is a row of ``COMMANDS``: an invocation, the files it needs in
the working directory, and the platform replies it reads before it would change
anything (``"METHOD /path": body``, or ``(status, body)``). A completeness test holds the
rows to the tree.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest
import typer
from dirty_equals import IsInt, IsStr, IsUUID
from typer.core import TyperGroup

from hud.cli.__main__ import app
from hud.eval import Task
from hud.version import __version__
from tests.conftest import write_files
from tests.fixtures.services import Reply

from .conftest import API_KEY

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import Hud, HudEnv
    from tests.fixtures.services import FakeServices

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"
IMPORT_ID = "66666666-6666-4666-8666-666666666666"
TASKSET_ID = "eeeeeeee-0000-4000-8000-000000000001"
PROJECT_ID = "aaaaaaaa-0000-4000-8000-000000000002"
MODEL_ID = "cccccccc-0000-4000-8000-000000000102"
TASKS_PY = (
    'from hud import Environment\nenv = Environment("example")\n'
    '@env.template(id="solve")\nasync def solve(n: int = 0):\n'
    '    yield f"solve {n}"\n    yield 1.0\n'
    "tasks = [solve(n=1)]\n"
)
ENVIRONMENT = {
    "env.py": 'from hud import Environment\nenv = Environment("example")\n',
    "Dockerfile.hud": "FROM python:3.12\n",
}
HARBOR = {"tasks/hello/task.toml": "version = '1.0'\n", "run-1/config.json": "{}"}
IMPORT = {
    "id": IMPORT_ID,
    "format": "harbor",
    "taskset_id": TASKSET_ID,
    "file": {"id": IMPORT_ID, "filename": "bundle.zip", "size_bytes": 1},
    "status": "running",
    "created_at": "2026-10-07T12:00:00Z",
}
MODEL = {
    "id": MODEL_ID,
    "name": "My Sonnet",
    "model_name": "team/my-sonnet",
    "sdk_agent_type": "claude",
    "is_trainable": True,
    "provider": {"name": "anthropic"},
}
CHECKS = [{"key": "failure_analysis", "subject_type": "trace", "title": "Failure Analysis"}]
FORMATS = [{"format": "harbor", "title": "Harbor", "max_input_bytes": 1024**3}]
NOT_FOUND = (404, {"detail": "Not found"})

#: name -> (invocation, files, platform replies read before any change)
COMMANDS: dict[str, tuple[list[str], dict[str, str], dict[str, Any]]] = {
    "init": (["init", "thing", "--preset", "blank"], {}, {}),
    "serve": (["serve", "missing.py"], {}, {}),
    "set": (["set", "HUD_EXAMPLE=1"], {}, {}),
    "version": (["version"], {}, {}),
    "task list": (["task", "list", "--source", "tasks.py"], {"tasks.py": TASKS_PY}, {}),
    "task start": (["task", "start", "0", "--source", "tasks.py"], {"tasks.py": TASKS_PY}, {}),
    "task grade": (
        ["task", "grade", "0", "--source", "tasks.py", "--answer", "x"],
        {"tasks.py": TASKS_PY},
        {},
    ),
    "deploy": (["deploy", "--no-env"], ENVIRONMENT, {}),
    "eval": (
        ["eval", "Demo", "openai_compatible", "-m", "scripted"],
        {},
        {
            "GET /v2/tasksets/by-name/Demo": {"taskset_id": TASKSET_ID, "name": "Demo"},
            f"GET /v2/tasksets/{TASKSET_ID}/export": {
                "name": "Demo",
                "tasks": [{"name": "one", "env": "demo", "scenario": "solve", "args": {}}],
            },
        },
    ),
    "project": (["project"], {}, {}),
    "project list": (["project", "list"], {}, {}),
    "project create": (["project", "create", "UI Evals"], {}, {}),
    "project use": (["project", "use", PROJECT_ID], {}, {}),
    "sync": (
        ["sync"],
        {"tasks.py": TASKS_PY, ".hud/config.json": json.dumps({"taskset_id": TASKSET_ID})},
        {
            f"GET /v2/tasksets/{TASKSET_ID}": {"id": TASKSET_ID, "name": "demo"},
            f"GET /v2/tasksets/{TASKSET_ID}/export": {"name": "demo", "tasks": []},
        },
    ),
    "sync tasks": (
        ["sync", "tasks", "demo", "tasks.py"],
        {"tasks.py": TASKS_PY},
        {"GET /v2/tasksets/by-name/demo": NOT_FOUND},
    ),
    "sync env": (["sync", "env", "example"], {}, {}),
    "harbor import": (
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2"],
        HARBOR,
        {
            "GET /v2/imports/formats": {"formats": FORMATS},
            "GET /v2/tasksets/by-name/tb2": NOT_FOUND,
        },
    ),
    "harbor imports": (["harbor", "imports"], {}, {}),
    "harbor report": (["harbor", "report", IMPORT_ID], {}, {}),
    "harbor cancel": (
        ["harbor", "cancel", IMPORT_ID],
        {},
        {f"GET /v2/imports/{IMPORT_ID}": IMPORT},
    ),
    "qa": (["qa"], {}, {}),
    "qa list": (["qa", "list"], {}, {}),
    "qa run": (
        ["qa", "run", "failure_analysis", TRACE_ID],
        {},
        {"GET /v2/qa/checks": {"checks": CHECKS}},
    ),
    "qa results": (["qa", "results", TRACE_ID], {}, {}),
    "jobs": (["jobs"], {}, {}),
    "jobs list": (["jobs", "list"], {}, {}),
    "jobs get": (["jobs", "get", JOB_ID], {}, {}),
    "jobs cancel": (["jobs", "cancel", JOB_ID], {}, {}),
    "cancel": (["cancel", JOB_ID], {}, {}),
    "trace get": (["trace", "get", TRACE_ID], {}, {}),
    "models list": (["models", "list"], {}, {}),
    "models fork": (["models", "fork", "claude-sonnet-4-6", "--name", "mine"], {}, {}),
    "models checkpoints": (["models", "checkpoints", "team/my-sonnet"], {}, {}),
    "models head": (
        ["models", "head", "team/my-sonnet", "--set", "ckpt-1"],
        {},
        {"GET /v2/models": {"items": [MODEL], "total": 1}},
    ),
}
#: Commands that never call the platform; each fails with its own invocation.
OFFLINE = {
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
#: Commands with ``--quiet``: (invocation, files, platform replies, the ids it prints)
LISTS: dict[str, tuple[list[str], dict[str, str], dict[str, Any], list[str]]] = {
    "task list": (
        ["task", "list", "--source", "tasks.py"],
        {"tasks.py": TASKS_PY.replace("[solve(n=1)]", "[solve(n=1), solve(n=2)]")},
        {},
        [Task(env="example", id="solve", args={"n": n}).slug for n in (1, 2)],
    ),
    "project list": (
        ["project", "list"],
        {},
        {"GET /v2/projects": {"items": [{"id": PROJECT_ID, "name": "p"}], "total": 1}},
        [PROJECT_ID],
    ),
    "models list": (
        ["models", "list"],
        {},
        {"GET /v2/models": {"items": [MODEL], "total": 1}},
        ["team/my-sonnet"],
    ),
    "models checkpoints": (
        ["models", "checkpoints", "team/my-sonnet"],
        {},
        {
            "GET /v2/models": {"items": [MODEL], "total": 1},
            f"GET /v2/models/{MODEL_ID}/checkpoints": [{"id": "ckpt-1"}, {"id": "ckpt-2"}],
        },
        ["ckpt-1", "ckpt-2"],
    ),
    "jobs": (
        ["jobs"],
        {},
        {"GET /v2/jobs": {"items": [{"id": "job-1"}, {"id": "job-2"}]}},
        ["job-1", "job-2"],
    ),
    "jobs list": (["jobs", "list"], {}, {"GET /v2/jobs": {"items": [{"id": "job-1"}]}}, ["job-1"]),
    "jobs get": (
        ["jobs", "get", JOB_ID],
        {},
        {f"GET /v2/jobs/{JOB_ID}/traces": {"items": [{"id": TRACE_ID}]}},
        [TRACE_ID],
    ),
    "qa": (["qa"], {}, {"GET /v2/qa/checks": {"checks": CHECKS}}, ["failure_analysis"]),
    "qa list": (
        ["qa", "list"],
        {},
        {"GET /v2/qa/checks": {"checks": CHECKS}},
        ["failure_analysis"],
    ),
    "harbor imports": (
        ["harbor", "imports"],
        {},
        {"GET /v2/imports": {"imports": [IMPORT], "total": 1}},
        [IMPORT_ID],
    ),
}


def tree(command: Any, path: tuple[str, ...] = ()) -> dict[str, Any]:
    """Every invocable command by path; a group that runs on its own counts as one."""
    group = isinstance(command, TyperGroup)
    runs = path and (not group or getattr(command, "invoke_without_command", False))
    found = {" ".join(path): command} if runs else {}
    if group:
        for name, sub in command.commands.items():
            found |= tree(sub, (*path, name))
    return found


TREE = tree(typer.main.get_command(app))
PLATFORM = sorted(set(COMMANDS) - set(OFFLINE))


def serve(services: FakeServices, replies: dict[str, Any]) -> None:
    """Answer each ``"METHOD /path"`` with its body, or with ``(status, body)``."""
    for route, reply in replies.items():
        method, path = route.split(" ", 1)
        status, body = reply if isinstance(reply, tuple) else (200, reply)
        services.route("api", method, path, status=status, json=body)


def answer_everything(services: FakeServices, reply: Reply) -> None:
    for method in ("GET", "POST", "PUT", "DELETE"):
        services.route("api", method, "/{rest:path}", reply)


def failing(name: str, hud: Hud, hud_env: HudEnv) -> list[str]:
    """An invocation of ``name`` that fails: offline, its own; otherwise signed out."""
    if name in OFFLINE:
        return OFFLINE[name]
    argv, files, _ = COMMANDS[name]
    write_files(hud.cwd, files)
    hud_env.set(HUD_API_KEY=None)
    return [*argv, *(["--yes"] if name in MUTATING else [])]


def state(hud: Hud, hud_env: HudEnv) -> dict[Path, bytes]:
    """Every file under the working directory and the user's ``~/.hud``."""
    return {
        path: path.read_bytes()
        for root in (hud.cwd, hud_env.home / ".hud")
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def writes(services: FakeServices) -> list[str]:
    return [f"{r.method} {r.path}" for r in services.requests("api") if r.method != "GET"]


def test_every_command_in_the_tree_is_a_row() -> None:
    assert sorted(COMMANDS) == sorted(TREE)
    assert set(OFFLINE) | MUTATING <= set(COMMANDS)


def test_every_command_with_quiet_is_a_list_row() -> None:
    quiet = {
        name
        for name, command in TREE.items()
        if any("--quiet" in getattr(param, "opts", []) for param in getattr(command, "params", []))
    }

    assert sorted(quiet) == sorted(LISTS)


def test_bare_hud_prints_help_and_exits_2(hud: Hud) -> None:
    result = hud()

    assert (result.exit_code, result.stderr) == (2, "")
    assert "Usage:" in result.stdout and "COMMAND" in result.stdout


def test_an_unknown_command_is_a_usage_error_on_stderr(hud: Hud) -> None:
    result = hud("nope")

    assert (result.exit_code, result.stdout) == (2, ""), result
    assert "nope" in result.stderr


@pytest.mark.parametrize(
    ("latest", "environment", "notified"),
    [("999.0.0", {}, True), ("999.0.0", {"CI": "1"}, False), (__version__, {}, False)],
    ids=["outdated", "outdated-in-ci", "current"],
)
def test_an_outdated_install_prints_an_upgrade_notice(
    latest: str, environment: dict[str, str], notified: bool, hud: Hud, hud_env: HudEnv
) -> None:
    """The notice reads PyPI's latest version from a fresh cache, so no request is made."""
    hud_env.set(HUD_SKIP_VERSION_CHECK=None, CI=None)
    cache = hud_env.home / ".hud" / ".cache" / "version_check.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"latest": latest, "checked_at": time.time()}))

    result = hud("version", env=environment)

    assert result.exit_code == 0, result
    assert ("999.0.0" in result.stderr) is notified


# ─── analytics ──────────────────────────────────────────────────────────


@pytest.fixture
def analytics(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    """CLI analytics on, posting to the telemetry service."""
    hud_env.set(HUD_CLI_ANALYTICS_ENABLED="1", CI=None)
    services.route("telemetry", "POST", "/sdk-events/cli", json={})
    return services


def events(services: FakeServices) -> list[dict[str, Any]]:
    return [
        e
        for body in services.bodies("telemetry", "POST", "/sdk-events/cli")
        for e in body["events"]
    ]


def event(name: str, exit_code: int, error: Any) -> dict[str, Any]:
    """The anonymous event an invocation of command ``name`` posts."""
    command, _, subcommand = name.partition(" ")
    return {
        "command": command,
        "subcommand": subcommand or None,
        "exit_code": exit_code,
        "error_class": error,
        "duration_ms": IsInt(ge=0),
        "cli_version": __version__,
        "python_version": IsStr(regex=r"3\.\d+\.\d+"),
        "os": sys.platform,
        "is_ci": False,
        "install_id": IsUUID,
    }


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_invocation_posts_one_anonymous_event_naming_its_command(
    name: str, analytics: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    result = hud(*failing(name, hud, hud_env))

    assert result.exit_code != 0
    assert events(analytics) == [event(name, result.exit_code, IsStr)]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["version"], [event("version", 0, None)]),
        (["secret-name", "arg"], [event("other", 2, "UsageError")]),
        (["--version"], []),
    ],
    ids=["a-command", "an-unknown-command-is-not-named", "the-version-flag"],
)
def test_what_is_not_a_command_posts_nothing_that_names_it(
    argv: list[str], expected: list[dict[str, Any]], analytics: FakeServices, hud: Hud
) -> None:
    hud(*argv)

    assert events(analytics) == expected
    assert "secret-name" not in json.dumps(events(analytics))


def test_the_install_id_is_created_once_and_announced_once(
    analytics: FakeServices, hud: Hud
) -> None:
    first = hud("version")
    second = hud("version")

    assert len({e["install_id"] for e in events(analytics)}) == 1
    assert "HUD_CLI_ANALYTICS_ENABLED=0" in first.stderr
    assert second.stderr == ""


def test_analytics_opt_out_posts_nothing(
    analytics: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_CLI_ANALYTICS_ENABLED="0")

    hud("version")

    assert analytics.requests("telemetry") == []


def test_a_stalled_analytics_endpoint_does_not_hold_up_the_command(
    analytics: FakeServices, hud: Hud
) -> None:
    started = time.monotonic()
    hud("version")
    answered = time.monotonic() - started
    analytics.route("telemetry", "POST", "/sdk-events/cli", Reply(json={}, delay=20))

    started = time.monotonic()
    result = hud("version")
    stalled = time.monotonic() - started

    assert result.exit_code == 0
    assert len(analytics.requests("telemetry")) == 2
    assert stalled - answered < 3


@pytest.mark.parametrize("name", [name for name in PLATFORM if name != "deploy"])
def test_an_interrupted_command_exits_130(
    name: str, analytics: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_API_KEY=API_KEY)
    argv, files, _ = COMMANDS[name]
    write_files(hud.cwd, files)
    answer_everything(analytics, Reply(json={}, delay=60))
    process = hud.start(*argv, *(["--yes"] if name in MUTATING else []))

    deadline = time.monotonic() + 30
    while not analytics.requests("api"):
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, f"hud {name} never reached the platform"
        time.sleep(0.05)
    process.send_signal(signal.SIGINT)
    process.communicate(timeout=30)

    assert process.returncode == 130
    assert events(analytics) == [event(name, 130, "KeyboardInterrupt")]


# ─── changes ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(MUTATING))
@pytest.mark.parametrize("flag", ["", "--dry-run"], ids=["unconfirmed", "dry-run"])
def test_a_mutating_command_changes_nothing_unless_confirmed(
    name: str, flag: str, signed_in: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    argv, files, replies = COMMANDS[name]
    write_files(hud.cwd, files)
    serve(signed_in, replies)
    before = state(hud, hud_env)

    result = hud(*argv, *filter(None, [flag]), "--json")

    if flag:
        assert (result.exit_code, result.json["dry_run"]) == (0, True), result
    else:
        assert (result.exit_code, result.json["error"]) == (2, "usage"), result
        assert "--yes" in result.json["suggestion"]
    assert writes(signed_in) == []
    assert state(hud, hud_env) == before


# ─── output ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_a_failure_is_one_error_document_under_json_and_stays_off_stdout_in_text(
    name: str, services: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    argv = failing(name, hud, hud_env)

    as_json = hud(*argv, "--json")
    as_text = hud(*argv)

    error = as_json.json
    assert as_json.exit_code == (2 if error["error"] == "usage" else 1), as_json
    assert error["message"] not in as_json.stderr, as_json
    assert (as_text.exit_code, as_text.stdout) == (as_json.exit_code, ""), as_text
    assert error["message"] in as_text.stderr, as_text
    assert ("Usage: " in as_text.stderr) is (error["error"] == "usage"), as_text


@pytest.mark.parametrize("name", PLATFORM)
def test_a_refusal_from_the_platform_is_permission_denied_for_every_command(
    name: str, services: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_API_KEY=API_KEY)
    argv, files, _ = COMMANDS[name]
    write_files(hud.cwd, files)
    answer_everything(services, Reply(status=403, json={"detail": "Refused"}))

    result = hud(*argv, *(["--yes"] if name in MUTATING else []), "--json")

    assert (result.exit_code, result.json["error"]) == (1, "permission_denied"), result


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, "permission_denied"),
        (404, "not_found"),
        (409, "conflict"),
        (429, "rate_limited"),
        (500, "server_error"),
    ],
)
def test_each_platform_status_maps_to_one_error_kind(
    status: int, error: str, signed_in: FakeServices, hud: Hud
) -> None:
    answer_everything(signed_in, Reply(status=status, json={"detail": "Refused"}))

    result = hud("trace", "get", TRACE_ID, "--json")

    assert (result.exit_code, result.json["error"]) == (1, error), result


def test_a_reply_that_is_not_json_is_a_failure(signed_in: FakeServices, hud: Hud) -> None:
    answer_everything(signed_in, Reply(body="<html>", content_type="text/html"))

    result = hud("jobs", "list", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "failure"), result


def test_a_transient_platform_failure_is_retried_and_the_command_succeeds(
    signed_in: FakeServices, hud: Hud
) -> None:
    listed = Reply(json={"items": [{"id": "job-1"}]})
    signed_in.route("api", "GET", "/v2/jobs", Reply(status=503), Reply(status=502), listed)

    result = hud("jobs", "list", "--quiet")

    assert (result.exit_code, result.stdout) == (0, "job-1\n"), result
    assert len(signed_in.requests("api", "GET", "/v2/jobs")) == 3


@pytest.mark.parametrize("name", sorted(LISTS))
def test_a_list_prints_one_id_per_line_under_quiet(
    name: str, signed_in: FakeServices, hud: Hud
) -> None:
    argv, files, replies, ids = LISTS[name]
    write_files(hud.cwd, files)
    serve(signed_in, replies)

    result = hud(*argv, "--quiet")

    assert (result.exit_code, result.stdout.split("\n")[:-1]) == (0, ids), result


SECRET = "s3cr3t-c0d3"


@pytest.mark.parametrize(
    ("argv", "files", "env"),
    [
        (["set", f"HUD_API_KEY={SECRET}"], {}, {}),
        (
            [
                *("deploy", "--secret", "id=ENV_TOKEN,env=ENV_TOKEN"),
                *("--secret", "id=FILE_TOKEN,src=token.txt", "--dry-run"),
            ],
            {**ENVIRONMENT, ".env": f"SECRET={SECRET}\n", "token.txt": SECRET},
            {"ENV_TOKEN": SECRET},
        ),
        (
            [
                *("eval", "tasks.py", "openai_compatible", "-m", "custom"),
                *("-c", f"api_key={SECRET}", "-c", "base_url=http://127.0.0.1:9", "--dry-run"),
            ],
            {"tasks.py": TASKS_PY},
            {},
        ),
    ],
    ids=["set", "deploy", "eval"],
)
@pytest.mark.parametrize("mode", [[], ["--json"]], ids=["text", "json"])
def test_a_secret_given_to_a_command_never_appears_in_its_output(
    argv: list[str],
    files: dict[str, str],
    env: dict[str, str],
    mode: list[str],
    signed_in: FakeServices,
    hud: Hud,
) -> None:
    write_files(hud.cwd, files)

    result = hud(*argv, *mode, env=env)

    assert SECRET not in result.stdout + result.stderr, result
