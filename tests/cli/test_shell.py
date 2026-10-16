"""The ``hud`` shell: help, version, ``hud set``, the error contract, analytics, notices."""

from __future__ import annotations

import json
import re
import signal
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsInt, IsStr, IsUUID
from dotenv import dotenv_values

from hud.version import __version__
from tests.support import Hud, Reply

from .conftest import API_KEY

if TYPE_CHECKING:
    from tests.support import FakeServices, HudEnv

JOB_ID = "03dd2a73-d3df-4d10-a54a-e3d87c2d530d"
TRACE_ID = "00000000-0000-4000-a000-000000000002"


def test_bare_hud_prints_help_and_exits_2(hud: Hud) -> None:
    result = hud()

    assert result.exit_code == 2
    assert result.stderr == ""
    # A wrapped description continues on a row whose name column is blank.
    rows = re.findall(r"^│ (\S+)", result.stdout, flags=re.MULTILINE)
    assert [row for row in rows if not row.startswith("-")] == [
        *("init", "serve", "deploy", "eval", "task", "project", "sync", "harbor"),
        *("qa", "jobs", "trace", "models", "set", "version"),
    ]


def test_version_prints_the_installed_version(hud: Hud) -> None:
    result = hud("version")

    assert (result.exit_code, result.stdout.split()[-1]) == (0, __version__), result


def test_version_json_names_the_package_and_version(hud: Hud) -> None:
    result = hud("version", "--json")

    assert (result.exit_code, result.json) == (0, {"name": "hud", "version": __version__})


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [
        pytest.param(["nope"], 2, id="an-unknown-command"),
        pytest.param(["jobs", "list"], 1, id="signed-out"),
    ],
)
def test_a_text_mode_failure_writes_only_to_stderr(
    hud: Hud, argv: list[str], exit_code: int
) -> None:
    result = hud(*argv)

    assert (result.exit_code, result.stdout) == (exit_code, ""), result
    assert result.stderr.strip()


@pytest.mark.parametrize(
    ("argv", "exit_code", "error"),
    [
        pytest.param(["set", "NOT_A_PAIR"], 2, "usage", id="a-malformed-argument"),
        pytest.param(["jobs", "list"], 1, "permission_denied", id="signed-out"),
    ],
)
def test_a_json_mode_failure_is_one_error_document_on_stdout(
    hud: Hud, argv: list[str], exit_code: int, error: str
) -> None:
    result = hud(*argv, "--json")

    assert (result.exit_code, result.json["error"], result.stderr) == (exit_code, error, ""), result


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
    ("status", "error"),
    [
        (404, "not_found"),
        (401, "permission_denied"),
        (403, "permission_denied"),
        (409, "conflict"),
        (429, "rate_limited"),
        (500, "server_error"),
    ],
)
def test_platform_errors_map_to_error_kinds(
    hud: Hud, services: FakeServices, hud_env: HudEnv, status: int, error: str
) -> None:
    hud_env.set(HUD_API_KEY=API_KEY)
    services.route("api", "GET", "/v2/jobs/{id}/traces", status=status, json={"detail": "No"})

    result = hud("jobs", "get", JOB_ID, "--json")

    assert (result.exit_code, result.json["error"]) == (1, error), result
    assert result.json["input"] == {"job_id": JOB_ID}


@pytest.mark.parametrize(
    ("latest", "argv", "environment", "notified"),
    [
        ("999.0.0", ["version"], {}, True),
        ("999.0.0", ["version"], {"CI": "1"}, False),
        (__version__, ["version"], {}, False),
    ],
)
def test_an_outdated_install_prints_an_upgrade_notice(
    hud: Hud,
    hud_env: HudEnv,
    latest: str,
    argv: list[str],
    environment: dict[str, str],
    notified: bool,
) -> None:
    """The notice reads PyPI's latest version from a fresh cache, so no request is made."""
    hud_env.set(HUD_SKIP_VERSION_CHECK=None, CI=None)
    cache = hud_env.home / ".hud" / ".cache" / "version_check.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"latest": latest, "checked_at": time.time()}))

    result = hud(*argv, env=environment)

    assert result.exit_code == 0, result
    assert ("999.0.0" in result.stderr) is notified


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
        (["jobs", "list", "--json"], None, [event("jobs", "list", 1, "HudAuthenticationError")]),
        (["set", "NOT_A_PAIR"], None, [event("set", None, 2, "CliError")]),
        (["secret-name", "arg"], None, [event("other", None, 2, "UsageError")]),
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
    assert "HUD_CLI_ANALYTICS_ENABLED=0" in first.stderr
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
    check = {"key": "failure_analysis", "subject_type": "trace", "title": "Failure Analysis"}
    analytics.route("api", "GET", "/v2/qa/checks", json={"checks": [check]})
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
