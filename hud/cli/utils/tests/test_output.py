"""CLI I/O contract: JSON stdout, exit codes, and confirmation policy."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import typer

from hud.cli.utils.output import (
    CliError,
    ExitCode,
    abort,
    confirm_or_abort,
    emit_json,
    emit_quiet,
    map_request_error,
    read_text_arg,
    resolve_output_mode,
    suppress_json_stdout,
    wants_json,
)
from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_json_flag() -> None:
    from hud.cli.utils.output import _JSON_REQUESTED, _JSON_SUPPRESSED

    _JSON_REQUESTED.set(False)
    _JSON_SUPPRESSED.set(False)


def test_wants_json_from_flag_and_output() -> None:
    assert wants_json(True) is True
    assert wants_json(False, "json") is True
    assert wants_json(False, "table") is False
    assert wants_json(False) is False


def test_wants_json_ignores_typer_option_defaults() -> None:
    """Direct command calls leave typer.Option objects in place; they are not True."""
    assert wants_json(typer.Option(False, "--json")) is False  # type: ignore[arg-type]


def test_resolve_output_mode_prefers_json_over_quiet() -> None:
    assert resolve_output_mode(json_output=True, quiet=True) == "json"
    assert resolve_output_mode(quiet=True) == "quiet"
    assert resolve_output_mode() == "table"


def test_resolve_output_mode_rejects_unknown_format() -> None:
    with pytest.raises(typer.Exit) as exc_info:
        resolve_output_mode(output="yaml")
    assert exc_info.value.exit_code == ExitCode.USAGE


def test_emit_json_goes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    emit_json({"id": "job-1", "count": 2})
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"id": "job-1", "count": 2}
    assert captured.err == ""


def test_emit_quiet_one_value_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    emit_quiet(["a", "b"])
    captured = capsys.readouterr()
    assert captured.out == "a\nb\n"
    assert captured.err == ""


def test_suppress_json_stdout_blocks_abort_json(capsys: pytest.CaptureFixture[str]) -> None:
    from hud.cli.utils.output import _JSON_REQUESTED

    _JSON_REQUESTED.set(True)
    with suppress_json_stdout(), pytest.raises(typer.Exit) as exc_info:
        abort(CliError(error="permission_denied", message="No HUD API key found"))
    assert exc_info.value.exit_code == ExitCode.PERMISSION
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error: No HUD API key found" in captured.err


def test_abort_writes_error_json_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        abort(
            CliError(
                error="not_found",
                message="Job missing",
                input={"job_id": "abc"},
                suggestion="hud jobs list",
            ),
            json_output=True,
        )
    assert exc_info.value.exit_code == ExitCode.NOT_FOUND
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["error"] == "not_found"
    assert payload["input"] == {"job_id": "abc"}
    assert payload["suggestion"] == "hud jobs list"
    assert "Error: Job missing" in captured.err
    assert "Hint: hud jobs list" in captured.err


def test_map_request_error_status_codes() -> None:
    assert map_request_error(HudRequestError("x", status_code=404)).exit_code == ExitCode.NOT_FOUND
    assert map_request_error(HudRequestError("x", status_code=403)).exit_code == ExitCode.PERMISSION
    assert map_request_error(HudRequestError("x", status_code=409)).exit_code == ExitCode.CONFLICT
    mapped = map_request_error(HudRequestError("x", status_code=429))
    assert mapped.transient is True
    assert mapped.error == "rate_limited"


def test_confirm_or_abort_noninteractive_requires_yes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hud.cli.utils.output.is_interactive", lambda: False)
    with pytest.raises(typer.Exit) as exc_info:
        confirm_or_abort("Proceed?")
    assert exc_info.value.exit_code == ExitCode.USAGE


def test_confirm_or_abort_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hud.cli.utils.output.is_interactive", lambda: False)
    confirm_or_abort("Proceed?", yes=True)


def test_read_text_arg_stdin_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "answer.txt"
    target.write_text("hello", encoding="utf-8")
    assert read_text_arg(str(target)) == "hello"

    monkeypatch.setattr("hud.cli.utils.output.sys.stdin.read", lambda: "from-stdin")
    assert read_text_arg("-") == "from-stdin"


def test_read_text_arg_missing_file_is_not_found() -> None:
    with pytest.raises(typer.Exit) as exc_info:
        read_text_arg("/definitely/missing/hud-cli-file.txt")
    assert exc_info.value.exit_code == ExitCode.NOT_FOUND
