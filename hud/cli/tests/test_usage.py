"""Tests for anonymous CLI usage events."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import typer

from hud.cli.app import app, recorded_invocation
from hud.utils.exceptions import HudException

if TYPE_CHECKING:
    from pathlib import Path


def _event(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: BaseException | None = None,
) -> dict[str, Any] | None:
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        httpx, "post", lambda _url, json=None, **_k: sent.append(json) or httpx.Response(204)
    )
    try:
        with recorded_invocation(argv, app):
            if error is not None:
                raise error
    except BaseException as exc:
        if exc is not error and type(exc) is not type(error):
            raise
    if not sent:
        return None
    (payload,) = sent
    (event,) = payload["events"]
    return event


@pytest.fixture(autouse=True)
def _analytics_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HUD_CLI_ANALYTICS_ENABLED", "1")
    monkeypatch.setenv("HUD_TELEMETRY_URL", "https://telemetry.example.test/v3/api")


def test_arguments_are_never_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(["hud", "eval", "tasks.py", "claude"], monkeypatch)
    assert event is not None
    assert event["command"] == "eval"
    assert event["subcommand"] is None


def test_registered_subcommands_are_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(["hud", "models", "list"], monkeypatch)
    assert event is not None
    assert (event["command"], event["subcommand"]) == ("models", "list")


def test_callback_group_positionals_are_never_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = _event(["hud", "trace", "8b1f2c3d4e5f"], monkeypatch)
    jobs = _event(["hud", "jobs", "0f9e8d7c"], monkeypatch)
    assert trace is not None and jobs is not None
    assert (trace["command"], trace["subcommand"]) == ("trace", None)
    assert (jobs["command"], jobs["subcommand"]) == ("jobs", None)


def test_jobs_verbs_are_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _event(["hud", "jobs", "list"], monkeypatch)["subcommand"] == "list"
    assert _event(["hud", "jobs", "cancel"], monkeypatch)["subcommand"] == "cancel"
    assert _event(["hud", "trace", "get"], monkeypatch)["subcommand"] == "get"
    assert _event(["hud", "qa", "list"], monkeypatch)["subcommand"] == "list"


def test_unregistered_command_is_other(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(["hud", "secret-name"], monkeypatch)
    assert event is not None
    assert event["command"] == "other"
    assert "secret-name" not in event.values()


def test_bare_invocation_is_help(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(["hud"], monkeypatch)
    assert event is not None
    assert event["command"] == "help"


def test_version_flag_is_not_an_event(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _event(["hud", "--version"], monkeypatch) is None
    assert _event(["hud", "--version", "--json"], monkeypatch) is None


def test_flags_are_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(["hud", "--verbose", "serve"], monkeypatch)
    assert event is not None
    assert event["command"] == "serve"


def test_typer_exit_from_hud_exception_names_the_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        raise HudException("boom")
    except HudException as exc:
        converted = typer.Exit(1)
        converted.__cause__ = exc
    event = _event(["hud", "eval"], monkeypatch, error=converted)
    assert event is not None
    assert (event["exit_code"], event["error_class"]) == (1, "HudException")


def test_plain_exit_has_no_error_class(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(["hud", "eval"], monkeypatch, error=typer.Exit(2))
    assert event is not None
    assert (event["exit_code"], event["error_class"]) == (2, None)


def test_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(["hud", "eval"], monkeypatch, error=KeyboardInterrupt())
    assert event is not None
    assert (event["exit_code"], event["error_class"]) == (130, "KeyboardInterrupt")


def test_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    event = _event(["hud", "eval"], monkeypatch, error=ValueError("x"))
    assert event is not None
    assert (event["exit_code"], event["error_class"]) == (1, "ValueError")


def test_install_id_created_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _event(["hud", "eval"], monkeypatch)
    second = _event(["hud", "eval"], monkeypatch)
    assert first is not None and second is not None
    assert first["install_id"] == second["install_id"]
    assert uuid.UUID(first["install_id"])
    assert capsys.readouterr().err.count("anonymous CLI usage") == 1


def test_opt_out_applies_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUD_CLI_ANALYTICS_ENABLED", "0")
    assert _event(["hud", "eval"], monkeypatch) is None


def test_command_error_propagates_when_opted_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUD_CLI_ANALYTICS_ENABLED", "0")
    with (
        pytest.raises(ValueError, match="boom"),
        recorded_invocation(["hud", "eval"], app),
    ):
        raise ValueError("boom")


def test_payload_is_the_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, json=None, **_k: sent.append((url, json)) or httpx.Response(204),
    )
    with recorded_invocation(["hud", "serve", "my_env.py"], app):
        pass
    (url, payload) = sent[0]
    assert url == "https://telemetry.example.test/v3/api/sdk-events/cli"
    (event,) = payload["events"]
    assert event["command"] == "serve"
    assert event["subcommand"] is None
    assert "my_env.py" not in str(payload)
    assert set(event) == {
        "command",
        "subcommand",
        "exit_code",
        "error_class",
        "duration_ms",
        "cli_version",
        "python_version",
        "os",
        "is_ci",
        "install_id",
    }
