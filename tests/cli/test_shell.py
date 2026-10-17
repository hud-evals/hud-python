"""The ``hud`` shell: help, an unknown command, ``hud set``, analytics policy, notices.

Contracts every command keeps live in ``test_every_command``.
"""

from __future__ import annotations

import json
import sys
import time
from typing import TYPE_CHECKING, Any

import pytest
from dirty_equals import IsInt, IsStr, IsUUID
from dotenv import dotenv_values

from hud.version import __version__
from tests.support import Hud, Reply

if TYPE_CHECKING:
    from tests.support import HudEnv, Platform


def test_bare_hud_prints_help_and_exits_2(hud: Hud) -> None:
    result = hud()

    assert (result.exit_code, result.stderr) == (2, "")
    assert "Usage:" in result.stdout and "COMMAND" in result.stdout


def test_an_unknown_command_is_a_usage_error_on_stderr(hud: Hud) -> None:
    result = hud("nope")

    assert (result.exit_code, result.stdout) == (2, ""), result
    assert "nope" in result.stderr


def test_set_persists_values_in_the_user_env_file(hud: Hud, hud_env: HudEnv) -> None:
    secret = "sk-secret # with 'quotes'"

    first = hud("set", "HUD_API_KEY=" + secret, "B=x # y", "--json")
    second = hud("set", "HUD_DEFAULT_PROJECT=example")

    path = hud_env.home / ".hud" / ".env"
    assert (first.exit_code, second.exit_code) == (0, 0), first
    assert dotenv_values(path, interpolate=False) == {
        "HUD_API_KEY": secret,
        "B": "x # y",
        "HUD_DEFAULT_PROJECT": "example",
    }


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
    ("argv", "expected"),
    [
        (["version"], [event("version", None, 0, None)]),
        (["secret-name", "arg"], [event("other", None, 2, "UsageError")]),
        (["--version"], []),
    ],
)
def test_each_invocation_posts_one_anonymous_event(
    hud: Hud,
    analytics: Platform,
    hud_env: HudEnv,
    argv: list[str],
    expected: list[dict[str, Any]],
) -> None:
    hud_env.set(HUD_API_KEY=None)

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
