"""What ``hud`` does around every command: the registry of commands, help and unknown
commands, the upgrade notice, anonymous analytics, and an interrupt."""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsInt, IsStr, IsUUID

from hud.version import __version__
from tests.support import Reply

from .commands import (
    EVERY_COMMAND,
    LOCAL_ONLY,
    MUTATING,
    OFFLINE,
    PLATFORM,
    READ_ONLY,
    TREE,
    answer_everything,
    failing,
    invocation,
)

if TYPE_CHECKING:
    from tests.support import Hud, HudEnv, Platform


def test_every_command_is_registered_and_classified() -> None:
    by_reach = [set(PLATFORM), set(OFFLINE)]
    by_change = [MUTATING, LOCAL_ONLY, READ_ONLY]

    for kinds in (by_reach, by_change):
        assert sorted(set(TREE) - set().union(*kinds)) == []
        assert sorted(set().union(*kinds) - set(TREE)) == []
        assert sum(len(kind) for kind in kinds) == len(TREE)


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
    [
        ("999.0.0", {}, True),
        ("999.0.0", {"CI": "1"}, False),
        (__version__, {}, False),
    ],
)
def test_an_outdated_install_prints_an_upgrade_notice(
    hud: Hud,
    hud_env: HudEnv,
    latest: str,
    environment: dict[str, str],
    notified: bool,
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


@pytest.mark.parametrize("name", EVERY_COMMAND)
def test_every_invocation_posts_one_anonymous_event_naming_its_command(
    hud: Hud, analytics: Platform, hud_env: HudEnv, name: str
) -> None:
    argv = failing(name, hud, analytics, hud_env)
    before = len(analytics.cli_events)

    result = hud(*argv)

    assert result.exit_code != 0
    assert analytics.cli_events[before:] == [event(name, result.exit_code, IsStr)]


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
    hud: Hud, analytics: Platform, argv: list[str], expected: list[dict[str, Any]]
) -> None:
    hud(*argv)

    assert analytics.cli_events == expected
    assert "secret-name" not in json.dumps(analytics.cli_events)


def test_the_install_id_is_created_once_and_announced_once(hud: Hud, analytics: Platform) -> None:
    first = hud("version")
    second = hud("version")

    assert len({event["install_id"] for event in analytics.cli_events}) == 1
    assert "HUD_CLI_ANALYTICS_ENABLED=0" in first.stderr
    assert second.stderr == ""


def test_analytics_opt_out_posts_nothing(hud: Hud, analytics: Platform, hud_env: HudEnv) -> None:
    hud_env.set(HUD_CLI_ANALYTICS_ENABLED="0")

    hud("version")

    assert analytics.services.requests("telemetry") == []


def test_a_stalled_analytics_endpoint_does_not_hold_up_the_command(
    hud: Hud, analytics: Platform
) -> None:
    analytics.services.route("telemetry", "POST", "/sdk-events/cli", Reply(json={}, delay=20))

    started = time.monotonic()
    result = hud("version")
    elapsed = time.monotonic() - started

    assert result.exit_code == 0
    assert len(analytics.services.requests("telemetry")) == 1
    # Without the bound the command would outlast the 20 s stall; with it, a few seconds.
    assert elapsed < 15, elapsed


# ─── interrupt ──────────────────────────────────────────────────────────


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
    assert analytics.cli_events[posted:] == [event(name, 130, "KeyboardInterrupt")]
