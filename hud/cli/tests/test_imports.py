"""Tests for ``hud.cli.imports``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import Mock
from uuid import UUID

from typer.testing import CliRunner

from hud.cli import ExitCode
from hud.cli.__main__ import app
from hud.integrations.harbor import HarborImportResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

DATA_FILE_ID = UUID("00000000-0000-4000-a000-000000000001")
JOB_ID = UUID("00000000-0000-4000-a000-000000000002")
TRACE_ID = UUID("00000000-0000-4000-a000-000000000003")


def _stub_import(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, object]:
    import_mock = Mock(
        return_value=HarborImportResult(
            data_file_id=DATA_FILE_ID,
            job_id=JOB_ID,
            trace_id=TRACE_ID,
        )
    )
    platform = object()
    monkeypatch.setattr("hud.cli.imports.PlatformClient.from_settings", lambda: platform)
    monkeypatch.setattr("hud.cli.imports.import_results", import_mock)
    return import_mock, platform


def test_hud_import_harbor_submits_archive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archive = tmp_path / "harbor.zip"
    archive.write_bytes(b"archive")
    import_mock, platform = _stub_import(monkeypatch)

    result = CliRunner().invoke(
        app,
        ["import", "harbor", str(archive), "--taskset", "terminal-bench", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "job_id": str(JOB_ID),
        "trace_id": str(TRACE_ID),
        "data_file_id": str(DATA_FILE_ID),
    }
    import_mock.assert_called_once_with(
        archive,
        taskset="terminal-bench",
        data_file_id=None,
        platform=platform,
    )


def test_hud_import_harbor_reuses_data_file(monkeypatch: pytest.MonkeyPatch) -> None:
    import_mock, platform = _stub_import(monkeypatch)

    result = CliRunner().invoke(
        app,
        ["import", "harbor", "--data-file-id", str(DATA_FILE_ID)],
    )

    assert result.exit_code == 0, result.output
    assert "Harbor import submitted" in result.output
    import_mock.assert_called_once_with(
        None,
        taskset=None,
        data_file_id=DATA_FILE_ID,
        platform=platform,
    )


def test_hud_import_harbor_requires_archive_or_data_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_mock, _ = _stub_import(monkeypatch)

    result = CliRunner().invoke(app, ["import", "harbor", "--json"])

    assert result.exit_code == ExitCode.USAGE, result.output
    assert json.loads(result.stdout)["error"] == "usage"
    import_mock.assert_not_called()
