"""Tests for anonymous CLI usage events."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest

from hud.cli.utils import usage

if TYPE_CHECKING:
    from pathlib import Path


class TestCommandTokens:
    @pytest.fixture(autouse=True)
    def _fresh_registry(self) -> None:
        """Rebuild the cached registry so earlier tests cannot poison it."""
        usage._registered_commands.cache_clear()

    def test_arguments_are_never_captured(self) -> None:
        """The second token of a plain command is user input and must not be sent."""
        assert usage._command_tokens(["hud", "eval", "tasks.py", "claude"]) == ("eval", None)

    def test_registered_subcommands_are_captured(self) -> None:
        assert usage._command_tokens(["hud", "models", "list"]) == ("models", "list")

    def test_callback_group_positionals_are_never_captured(self) -> None:
        """``hud trace <id>`` and ``hud jobs <id>`` take user data, not subcommands."""
        assert usage._command_tokens(["hud", "trace", "8b1f2c3d4e5f"]) == ("trace", None)
        assert usage._command_tokens(["hud", "jobs", "0f9e8d7c"]) == ("jobs", None)

    def test_unregistered_command_is_other(self) -> None:
        """A token that is not a registered command is never sent verbatim."""
        assert usage._command_tokens(["hud", "secret-name"]) == ("other", None)

    def test_bare_invocation_is_help(self) -> None:
        assert usage._command_tokens(["hud"]) == ("help", None)

    def test_flags_are_skipped(self) -> None:
        assert usage._command_tokens(["hud", "--verbose", "serve"]) == ("serve", None)


class TestClassify:
    def test_typer_exit_from_hud_exception_names_the_cause(self) -> None:
        """The CLI converts HudException via ``raise typer.Exit(1) from e``."""
        import typer

        from hud.utils.exceptions import HudException

        try:
            try:
                raise HudException("boom")
            except HudException as e:
                raise typer.Exit(1) from e
        except typer.Exit as converted:
            assert usage._classify(converted) == (1, "HudException")

    def test_plain_exit_has_no_error_class(self) -> None:
        import typer

        assert usage._classify(typer.Exit(2)) == (2, None)

    def test_keyboard_interrupt(self) -> None:
        assert usage._classify(KeyboardInterrupt()) == (130, "KeyboardInterrupt")

    def test_unexpected_exception(self) -> None:
        assert usage._classify(ValueError("x")) == (1, "ValueError")


class TestInstallId:
    def test_created_once_and_persisted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """First call creates an id and prints the notice; later calls reuse it silently."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        first = usage._install_id()
        second = usage._install_id()

        assert first == second
        assert uuid.UUID(first)
        captured = capsys.readouterr()
        assert captured.err.count("anonymous usage data") == 1


class TestRecordInvocation:
    def test_opt_out_applies_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The opt-out is re-read at send time, so it applies to the very
        invocation that set it — even though the import-time settings
        singleton still says enabled."""
        from hud.settings import settings

        monkeypatch.setattr(settings, "telemetry_enabled", True)
        monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "0")

        assert (
            usage.record_invocation(["hud", "eval"], exit_code=0, error_class=None, duration_ms=10)
            is None
        )

    def test_payload_is_the_allowlist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sent payload holds command facts only — no argv, paths, or messages."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "1")
        monkeypatch.setenv("HUD_API_URL", "https://api.example.test")
        sent: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(usage, "_post", lambda url, payload: sent.append((url, payload)))

        thread = usage.record_invocation(
            ["hud", "serve", "my_env.py"],
            exit_code=1,
            error_class="HudException",
            duration_ms=42,
        )

        assert thread is not None
        thread.join(timeout=5)
        (url, payload) = sent[0]
        assert url == "https://api.example.test/v2/sdk-events/cli"
        (event,) = payload["events"]
        assert event["command"] == "serve"
        assert event["subcommand"] is None  # my_env.py must not appear
        assert event["exit_code"] == 1
        assert event["error_class"] == "HudException"
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
