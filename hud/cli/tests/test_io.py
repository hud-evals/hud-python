"""CLI I/O contract: JSON stdout, exit codes, and confirmation policy."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from hud.cli.io import (
    CliError,
    ExitCode,
    confirm_or_abort,
    emit_error,
    emit_json,
    emit_quiet,
    json_object,
    map_request_error,
    read_text_arg,
    report,
)
from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from pathlib import Path


def test_json_object_names_the_flag() -> None:
    assert json_object('{"image": "x"}', option="--payload") == {"image": "x"}
    with pytest.raises(CliError) as exc_info:
        json_object("[]", option="--payload")
    assert exc_info.value.message == "--payload must be a JSON object"
    assert exc_info.value.input == {"payload": "[]"}


def test_emit_json_goes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    emit_json({"id": "job-1", "count": 2})
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"id": "job-1", "count": 2}
    assert captured.err == ""


def test_report_json_and_human_use_the_same_payload(capsys: pytest.CaptureFixture[str]) -> None:
    payload = {"path": "/tmp/.hud/.env", "keys": ["HUD_API_KEY"]}
    seen: list[object] = []
    report(payload, json_output=True, render=seen.append)
    assert json.loads(capsys.readouterr().out) == payload
    assert seen == []
    report(payload, json_output=False, render=seen.append)
    assert seen == [payload]


def test_emit_quiet_one_value_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    emit_quiet(["a", "b"])
    captured = capsys.readouterr()
    assert captured.out == "a\nb\n"
    assert captured.err == ""


def test_output_mode_does_not_leak_between_invocations() -> None:
    from typer.testing import CliRunner

    from hud.cli import app

    runner = CliRunner()
    first = runner.invoke(app, ["version", "--json"])
    second = runner.invoke(app, ["version"])
    flag = runner.invoke(app, ["--version"])
    assert json.loads(first.stdout)["name"] == "hud"
    assert second.stdout.startswith("HUD CLI version:")
    assert flag.exit_code == 0
    assert "HUD CLI version:" in flag.stdout
    unexpected = runner.invoke(app, ["--json", "version"])
    assert unexpected.exit_code != 0


def test_map_request_error_status_codes() -> None:
    not_found = map_request_error(HudRequestError("x", status_code=404))
    assert not_found.error == "not_found"
    assert not_found.exit_code == ExitCode.FAILURE
    permission = map_request_error(HudRequestError("x", status_code=403))
    assert permission.error == "permission_denied"
    assert permission.exit_code == ExitCode.FAILURE
    conflict = map_request_error(HudRequestError("x", status_code=409))
    assert conflict.error == "conflict"
    assert conflict.exit_code == ExitCode.FAILURE
    mapped = map_request_error(HudRequestError("x", status_code=429))
    assert mapped.error == "rate_limited"


def test_emit_error_json_writes_only_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    emit_error(
        CliError(error="not_found", message="missing job", suggestion="Check the id."),
        json_output=True,
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"] == "not_found"
    assert captured.err == ""


def test_emit_error_text_writes_only_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    emit_error(CliError(error="not_found", message="missing job", suggestion="Check the id."))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error: missing job" in captured.err
    assert "Hint: Check the id." in captured.err


def test_confirm_or_abort_noninteractive_requires_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(CliError) as exc_info:
        confirm_or_abort("Proceed?")
    assert exc_info.value.exit_code == ExitCode.USAGE


def test_confirm_or_abort_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    confirm_or_abort("Proceed?", yes=True)


def test_read_text_arg_stdin_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "answer.txt"
    target.write_text("hello", encoding="utf-8")
    assert read_text_arg(str(target)) == "hello"

    monkeypatch.setattr("hud.cli.io.sys.stdin.read", lambda: "from-stdin")
    assert read_text_arg("-") == "from-stdin"


def test_read_text_arg_missing_file_is_not_found() -> None:
    with pytest.raises(CliError) as exc_info:
        read_text_arg("/definitely/missing/hud-cli-file.txt")
    assert exc_info.value.error == "not_found"
    assert exc_info.value.exit_code == ExitCode.FAILURE
