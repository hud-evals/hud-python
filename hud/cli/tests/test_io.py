"""CLI I/O contract: JSON stdout, exit codes, and confirmation policy."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from hud.cli.app import (
    CLI,
    CliError,
    ExitCode,
)
from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from pathlib import Path


def test_json_object_names_the_flag() -> None:
    assert CLI.json_object('{"image": "x"}', option="--payload") == {"image": "x"}
    with pytest.raises(CliError) as exc_info:
        CLI.json_object("[]", option="--payload")
    assert exc_info.value.message == "--payload must be a JSON object"
    assert exc_info.value.input == {"payload": "[]"}


def test_output_mode_does_not_leak_between_invocations() -> None:
    from typer.testing import CliRunner

    from hud.cli.app import app

    runner = CliRunner()
    first = runner.invoke(app, ["version", "--json"])
    second = runner.invoke(app, ["version"])
    flag = runner.invoke(app, ["--version"])
    assert json.loads(first.stdout)["name"] == "hud"
    assert "HUD CLI version:" not in first.stdout
    assert second.stdout.startswith("HUD CLI version:")
    assert flag.exit_code == 0
    assert "HUD CLI version:" in flag.stdout
    unexpected = runner.invoke(app, ["--json", "version"])
    assert unexpected.exit_code != 0


def test_map_request_error_status_codes() -> None:
    not_found = CliError.from_http(HudRequestError("x", status_code=404))
    assert not_found.error == "not_found"
    assert not_found.exit_code == ExitCode.FAILURE
    permission = CliError.from_http(HudRequestError("x", status_code=403))
    assert permission.error == "permission_denied"
    assert permission.exit_code == ExitCode.FAILURE
    conflict = CliError.from_http(HudRequestError("x", status_code=409))
    assert conflict.error == "conflict"
    assert conflict.exit_code == ExitCode.FAILURE
    mapped = CliError.from_http(HudRequestError("x", status_code=429))
    assert mapped.error == "rate_limited"


def test_failure_text_writes_only_stderr() -> None:
    from typer.testing import CliRunner

    from hud.cli.app import app

    result = CliRunner().invoke(app, ["set", "NOT_A_PAIR"])
    assert result.exit_code == ExitCode.USAGE
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert "Hint:" in result.stderr


def test_confirm_or_abort_noninteractive_requires_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(CliError) as exc_info:
        CLI.confirm_or_abort("Proceed?")
    assert exc_info.value.exit_code == ExitCode.USAGE


def test_confirm_or_abort_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    CLI.confirm_or_abort("Proceed?", yes=True)


def test_read_text_stdin_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "answer.txt"
    target.write_text("hello", encoding="utf-8")
    assert CLI.read_text(str(target)) == "hello"

    monkeypatch.setattr("hud.cli.app.sys.stdin.read", lambda: "from-stdin")
    assert CLI.read_text("-") == "from-stdin"


def test_read_text_missing_file_is_not_found() -> None:
    with pytest.raises(CliError) as exc_info:
        CLI.read_text("/definitely/missing/hud-cli-file.txt")
    assert exc_info.value.error == "not_found"
    assert exc_info.value.exit_code == ExitCode.FAILURE
