"""What every ``hud`` command does alike: anonymous analytics, an interrupt, asking before it
changes anything, the error document a failure prints, the error kind a platform status
maps to, retrying a transient failure, one id per line under ``--quiet``, and never
echoing a secret it was given.

The CLI owns these contracts once, so each table holds one row per distinct way a command
reaches them, not one row per command.
"""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsInt, IsStr, IsUUID

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
TASKSET_ID = "eeeeeeee-0000-4000-8000-000000000001"
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
MODEL = {
    "id": "cccccccc-0000-4000-8000-000000000102",
    "name": "My Sonnet",
    "model_name": "team/my-sonnet",
    "sdk_agent_type": "claude",
    "is_trainable": True,
    "provider": {"name": "anthropic"},
}

#: name -> (an invocation that fails when signed out, the files it needs)
FAILING: dict[str, tuple[list[str], dict[str, str]]] = {
    "version": (["version", "--no-such-option"], {}),  # rejected while parsing
    "set": (["set", "NOT_A_PAIR"], {}),  # the command raises a usage error
    "task list": (["task", "list", "--source", "missing.json"], {}),  # a missing file
    "jobs": (["jobs"], {}),  # a group run without a subcommand
    "jobs list": (["jobs", "list"], {}),  # a platform read
    "deploy": (["deploy", "--no-env", "--yes"], ENVIRONMENT),  # an async command
}
#: name -> (invocation, files, platform replies read before any change)
MUTATING: dict[str, tuple[list[str], dict[str, str], dict[str, Any]]] = {
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
    # The group forwards --yes and --dry-run to ``sync tasks``.
    "sync": (
        ["sync"],
        {"tasks.py": TASKS_PY, ".hud/config.json": json.dumps({"taskset_id": TASKSET_ID})},
        {
            f"GET /v2/tasksets/{TASKSET_ID}": {"id": TASKSET_ID, "name": "demo"},
            f"GET /v2/tasksets/{TASKSET_ID}/export": {"name": "demo", "tasks": []},
        },
    ),
    # Packs the bundle before it asks.
    "harbor import": (
        ["harbor", "import", "tasks", "run-1", "--taskset", "tb2"],
        {"tasks/hello/task.toml": "version = '1.0'\n", "run-1/config.json": "{}"},
        {
            "GET /v2/imports/formats": {
                "formats": [{"format": "harbor", "title": "Harbor", "max_input_bytes": 1024**3}]
            },
            "GET /v2/tasksets/by-name/tb2": (404, {"detail": "Not found"}),
        },
    ),
    # Answers --dry-run before it reaches the platform.
    "jobs cancel": (["jobs", "cancel", JOB_ID], {}, {}),
    # Asks only when --set changes the head.
    "models head": (
        ["models", "head", "team/my-sonnet", "--set", "ckpt-1"],
        {},
        {"GET /v2/models": {"items": [MODEL], "total": 1}},
    ),
}
#: name -> (invocation, files, platform replies, the ids it prints under ``--quiet``)
LISTS: dict[str, tuple[list[str], dict[str, str], dict[str, Any], list[str]]] = {
    "task list": (
        ["task", "list", "--source", "tasks.py"],
        {"tasks.py": TASKS_PY.replace("[solve(n=1)]", "[solve(n=1), solve(n=2)]")},
        {},
        [Task(env="example", id="solve", args={"n": n}).slug for n in (1, 2)],
    ),
    "jobs": (
        ["jobs"],
        {},
        {"GET /v2/jobs": {"items": [{"id": "job-1"}, {"id": "job-2"}]}},
        ["job-1", "job-2"],
    ),
}


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
    """The invocation of ``name`` that fails, run signed out."""
    argv, files = FAILING[name]
    write_files(hud.cwd, files)
    hud_env.set(HUD_API_KEY=None)
    return argv


def state(hud: Hud, hud_env: HudEnv) -> dict[Path, bytes]:
    """Every file under the working directory and the user's ``~/.hud``."""
    return {
        path: path.read_bytes()
        for root in (hud.cwd, hud_env.home / ".hud")
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


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


def test_an_interrupted_command_exits_130(
    analytics: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_API_KEY=API_KEY)
    answer_everything(analytics, Reply(json={}, delay=60))
    process = hud.start("jobs", "list")

    deadline = time.monotonic() + 30
    while not analytics.requests("api"):
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, "hud jobs list never reached the platform"
        time.sleep(0.05)
    process.send_signal(signal.SIGINT)
    process.communicate(timeout=30)

    assert process.returncode == 130
    assert events(analytics) == [event("jobs list", 130, "KeyboardInterrupt")]


# ─── changes ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(MUTATING))
@pytest.mark.parametrize("flag", ["", "--dry-run"], ids=["unconfirmed", "dry-run"])
def test_a_mutating_command_changes_nothing_unless_confirmed(
    name: str, flag: str, signed_in: FakeServices, hud_env: HudEnv, hud: Hud
) -> None:
    argv, files, replies = MUTATING[name]
    write_files(hud.cwd, files)
    serve(signed_in, replies)
    before = state(hud, hud_env)

    result = hud(*argv, *filter(None, [flag]), "--json")

    if flag:
        assert (result.exit_code, result.json["dry_run"]) == (0, True), result
    else:
        assert (result.exit_code, result.json["error"]) == (2, "usage"), result
        assert "--yes" in result.json["suggestion"]
    assert [f"{r.method} {r.path}" for r in signed_in.requests("api") if r.method != "GET"] == []
    assert state(hud, hud_env) == before


# ─── output ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(FAILING))
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
