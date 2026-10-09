"""The ``hud`` shell: help, version, ``hud set``, the error contract, analytics, notices."""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsInt, IsStr, IsUUID
from dotenv import dotenv_values
from inline_snapshot import snapshot

from hud.version import __version__
from tests.harness import Reply, words

from .conftest import API_KEY

if TYPE_CHECKING:
    from tests.harness import FakeServices, Hud, HudEnv

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"


def test_bare_hud_prints_help_and_exits_2(hud: Hud) -> None:
    result = hud()

    assert result.exit_code == 2
    assert result.stderr == ""
    commands = result.lines[result.lines.index("Commands") + 1 :]
    assert [line.split()[0] for line in commands] == [
        *("init", "serve", "deploy", "eval", "task", "project", "sync"),
        *("qa", "jobs", "trace", "models", "set", "version"),
    ]


def test_root_help_lists_the_public_commands_in_order(hud: Hud) -> None:
    result = hud("--help")

    assert result.exit_code == 0
    assert result.stdout == hud().stdout


@pytest.mark.parametrize(
    ("group", "verbs"),
    [
        ("task", ["list", "start", "grade"]),
        ("project", ["list", "create", "use"]),
        ("sync", ["tasks", "env"]),
        ("qa", ["list", "run", "results"]),
        ("jobs", ["list", "get", "cancel"]),
        ("trace", ["get"]),
        ("models", ["list", "fork", "checkpoints", "head"]),
    ],
)
def test_group_help_lists_its_verbs(hud: Hud, group: str, verbs: list[str]) -> None:
    result = hud(group, "--help")

    assert result.exit_code == 0, result
    listed = [line.split()[1] for line in result.stdout.splitlines() if line.startswith("│ ")]
    assert set(verbs) <= set(listed)


@pytest.mark.parametrize(
    ("argv", "exit_code", "stdout", "stderr"),
    [
        (["--version"], 0, f"HUD CLI version: {__version__}\n", ""),
        (["version"], 0, f"HUD CLI version: {__version__}\n", ""),
        (["version", "--json"], 0, f'{{\n  "name": "hud",\n  "version": "{__version__}"\n}}\n', ""),
        (
            ["nope"],
            2,
            "",
            "Usage: python -m hud.cli COMMAND\nError: No such command 'nope'.\n",
        ),
        (
            ["auth", "--help"],
            2,
            "",
            "Usage: python -m hud.cli COMMAND\nError: No such command 'auth'.\n",
        ),
        (
            ["--json", "version"],
            2,
            "",
            snapshot("""\
Usage: python -m hud.cli COMMAND
╭─ Error ──────────────────────────────────────────────────────────────────────────────────────╮
│ No such option: --json (Possible options: --version)                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯
"""),
        ),
        (
            ["set", "NOT_A_PAIR"],
            2,
            "",
            "Usage: python -m hud.cli COMMAND\n"
            "Error: Invalid assignment (expected KEY=VALUE): NOT_A_PAIR\n"
            "Hint: Pass one or more KEY=VALUE pairs.\n",
        ),
        (
            ["set", "NOT_A_PAIR", "--json"],
            2,
            snapshot("""\
{
  "error": "usage",
  "message": "Invalid assignment (expected KEY=VALUE): NOT_A_PAIR",
  "input": {
    "assignment": "NOT_A_PAIR"
  },
  "suggestion": "Pass one or more KEY=VALUE pairs."
}
"""),
            "",
        ),
        (
            ["jobs", "list", "--json"],
            1,
            snapshot("""\
{
  "error": "permission_denied",
  "message": "HUD_API_KEY is required",
  "suggestion": "Run 'hud set HUD_API_KEY=your-key-here'."
}
"""),
            "",
        ),
        (
            ["jobs", "list"],
            1,
            "",
            "Error: HUD_API_KEY is required\nHint: Run 'hud set HUD_API_KEY=your-key-here'.\n",
        ),
    ],
)
def test_output_contract_per_mode(
    hud: Hud, argv: list[str], exit_code: int, stdout: str, stderr: str
) -> None:
    """Text mode writes failures only to stderr; ``--json`` writes one document to stdout."""
    result = hud(*argv)

    assert (result.exit_code, result.stdout, words(result.stderr)) == (
        exit_code,
        stdout,
        words(stderr),
    )


def test_set_persists_values_without_echoing_them(hud: Hud, hud_env: HudEnv) -> None:
    secret = "sk-secret # with 'quotes'"

    first = hud("set", "HUD_API_KEY=" + secret, "B=x # y", "--json")
    second = hud("set", "HUD_DEFAULT_PROJECT=example")

    path = hud_env.home / ".hud" / ".env"
    assert first.exit_code == 0, first
    assert first.json == {"path": str(path), "keys": ["HUD_API_KEY", "B"]}
    assert second.exit_code == 0, second
    assert secret not in first.stdout + first.stderr + second.stdout + second.stderr
    assert dotenv_values(path, interpolate=False) == {
        "HUD_API_KEY": secret,
        "B": "x # y",
        "HUD_DEFAULT_PROJECT": "example",
    }


@pytest.mark.parametrize(
    ("reply", "document"),
    [
        (
            Reply(status=404, json={"detail": "Job missing"}),
            snapshot(
                {
                    "error": "not_found",
                    "message": "Request failed: Job missing",
                    "input": {"job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"},
                    "suggestion": "Check the job id, or list existing ones.",
                }
            ),
        ),
        (
            Reply(status=401, json={"detail": "Bad key"}),
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "Request failed: Bad key",
                    "input": {"job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"},
                    "suggestion": "Check that this API key can access the resource.",
                }
            ),
        ),
        (
            Reply(status=403, json={"detail": "Not your team"}),
            snapshot(
                {
                    "error": "permission_denied",
                    "message": "Request failed: Not your team",
                    "input": {"job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"},
                    "suggestion": "Check that this API key can access the resource.",
                }
            ),
        ),
        (
            Reply(status=409, json={"detail": "Busy"}),
            snapshot(
                {
                    "error": "conflict",
                    "message": "Request failed: Busy",
                    "input": {"job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"},
                }
            ),
        ),
        (
            Reply(status=429, json={"detail": "Slow down"}),
            snapshot(
                {
                    "error": "rate_limited",
                    "message": "Request failed: Slow down",
                    "input": {"job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"},
                    "suggestion": "Retry after a short delay.",
                }
            ),
        ),
        (
            Reply(status=500, json={"detail": "Boom"}),
            snapshot(
                {
                    "error": "server_error",
                    "message": "Request failed: Boom",
                    "input": {"job_id": "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"},
                    "suggestion": "Retry; this error is often transient.",
                }
            ),
        ),
    ],
)
def test_platform_errors_map_to_error_documents(
    hud: Hud, services: FakeServices, hud_env: HudEnv, reply: Reply, document: dict[str, Any]
) -> None:
    hud_env.set(HUD_API_KEY=API_KEY)
    services.route("api", "GET", "/v2/jobs/{id}/traces", reply)

    result = hud("jobs", "get", JOB_ID, "--json")

    assert result.exit_code == 1
    assert result.json == document


def test_a_deprecated_endpoint_warns_once_per_process(
    hud: Hud, services: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY=API_KEY)
    deprecated = {
        "Deprecation": "@1767225600",
        "Sunset": "Wed, 01 Jul 2026 00:00:00 GMT",
        "Link": '<https://api.example/v3/projects>; rel="successor-version"',
    }
    project = {"id": JOB_ID, "name": "demo", "capabilities": {"create": True}}
    services.route(
        "api",
        "GET",
        "/v2/projects",
        Reply(json={"items": [project], "total": 2}, headers=deprecated),
        Reply(json={"items": [{**project, "id": TRACE_ID}], "total": 2}, headers=deprecated),
    )

    result = hud("project", "list", "--quiet")

    assert result.exit_code == 0, result
    assert result.stdout == f"{JOB_ID}\n{TRACE_ID}\n"
    assert result.stderr == snapshot("""\
⚠ GET /api/v2/projects is deprecated by the HUD API and stops being served on Wed, 01 Jul 2026 \n\
00:00:00 GMT; its replacement is https://api.example/v3/projects. If a hud command printed this,
upgrade hud.
""")


@pytest.mark.parametrize(
    ("latest", "argv", "environment", "notice"),
    [
        (
            "999.0.0",
            ["version"],
            {},
            f"A new version of hud is available: 999.0.0 (current: {__version__})\n"
            "Run: uv sync --upgrade-package hud\n",
        ),
        ("999.0.0", ["version"], {"CI": "1"}, ""),
        ("999.0.0", ["version"], {"HUD_SKIP_VERSION_CHECK": "1"}, ""),
        ("999.0.0", ["--version"], {}, ""),
        ("999.0.0", ["--help"], {}, ""),
        (__version__, ["version"], {}, ""),
    ],
)
def test_an_outdated_install_prints_an_upgrade_notice(
    hud: Hud,
    hud_env: HudEnv,
    latest: str,
    argv: list[str],
    environment: dict[str, str],
    notice: str,
) -> None:
    """The notice reads PyPI's latest version from a fresh cache, so no request is made."""
    hud_env.set(HUD_SKIP_VERSION_CHECK=None, CI=None)
    cache = hud_env.home / ".hud" / ".cache" / "version_check.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"latest": latest, "checked_at": time.time()}))

    result = hud(*argv, env=environment)

    assert result.exit_code == 0, result
    assert result.stderr == notice


# ─── analytics ──────────────────────────────────────────────────────────


@pytest.fixture
def analytics(services: FakeServices, hud_env: HudEnv) -> FakeServices:
    """CLI analytics on, posting to the fake telemetry service."""
    hud_env.set(HUD_CLI_ANALYTICS_ENABLED="1", CI=None)
    services.route("telemetry", "POST", "/sdk-events/cli", json={})
    return services


def event(
    command: str, subcommand: str | None, exit_code: int, error: str | None
) -> dict[str, Any]:
    return {
        "command": command,
        "subcommand": subcommand,
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
    ("argv", "key", "expected"),
    [
        (["version"], None, [event("version", None, 0, None)]),
        (["--json", "version"], None, [event("version", None, 2, "NoSuchOption")]),
        (["jobs", "list", "--json"], None, [event("jobs", "list", 1, "HudAuthenticationError")]),
        (["jobs", "list"], API_KEY, [event("jobs", "list", 1, "HudRequestError")]),
        (["set", "NOT_A_PAIR"], None, [event("set", None, 2, "CliError")]),
        (["jobs", "list", "--limit", "many"], None, [event("jobs", "list", 2, "BadParameter")]),
        (["secret-name", "arg"], None, [event("other", None, 2, "UsageError")]),
        (["trace", "8b1f2c3d4e5f"], None, [event("trace", None, 2, "UsageError")]),
        ([], None, [event("help", None, 2, None)]),
        (["--version"], None, []),
    ],
)
def test_each_invocation_posts_one_anonymous_event(
    hud: Hud,
    analytics: FakeServices,
    hud_env: HudEnv,
    argv: list[str],
    key: str | None,
    expected: list[dict[str, Any]],
) -> None:
    hud_env.set(HUD_API_KEY=key)
    analytics.route("api", "GET", "/v2/jobs", status=403, json={"detail": "Not your team"})

    hud(*argv)

    posted = [body["events"] for body in analytics.bodies("telemetry", "POST", "/sdk-events/cli")]
    assert [event for events in posted for event in events] == expected
    assert "secret-name" not in json.dumps(posted)


def test_the_install_id_is_created_once_and_announced_once(
    hud: Hud, analytics: FakeServices
) -> None:
    first = hud("version")
    second = hud("version")

    ids = [
        body["events"][0]["install_id"]
        for body in analytics.bodies("telemetry", "POST", "/sdk-events/cli")
    ]
    assert len(set(ids)) == 1
    assert first.stderr == snapshot(
        "hud collects anonymous CLI usage. Disable: hud set HUD_CLI_ANALYTICS_ENABLED=0\n"
    )
    assert second.stderr == ""


def test_analytics_opt_out_posts_nothing(
    hud: Hud, analytics: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_CLI_ANALYTICS_ENABLED="0")

    hud("version")

    assert analytics.requests("telemetry") == []


def test_a_stalled_analytics_endpoint_does_not_hold_up_the_command(
    hud: Hud, analytics: FakeServices
) -> None:
    analytics.route("telemetry", "POST", "/sdk-events/cli", Reply(json={}, delay=20))

    started = time.monotonic()
    result = hud("version")
    elapsed = time.monotonic() - started

    assert result.exit_code == 0
    assert len(analytics.requests("telemetry")) == 1
    # Without the bound the command would outlast the 20 s stall; with it, a few seconds.
    assert elapsed < 15, elapsed


def test_an_interrupted_command_reports_exit_130(
    hud: Hud, analytics: FakeServices, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY=API_KEY)
    pending = {"subject_trace_id": TRACE_ID, "check_key": "failure_analysis", "status": "queued"}
    analytics.route("api", "POST", "/v2/qa/runs", json={"results": [pending]})
    process = hud.start("qa", "run", "failure_analysis", TRACE_ID)

    deadline = time.monotonic() + 30
    while not analytics.requests("api", "POST", "/v2/qa/runs"):
        assert time.monotonic() < deadline, "hud qa run never posted the run"
        time.sleep(0.05)
    process.send_signal(signal.SIGINT)
    process.communicate(timeout=30)

    assert process.returncode == 130
    (body,) = analytics.bodies("telemetry", "POST", "/sdk-events/cli")
    assert body["events"] == [event("qa", "run", 130, "KeyboardInterrupt")]
