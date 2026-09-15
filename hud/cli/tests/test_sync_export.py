"""Focused behavior for ``hud sync`` commands."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import hud.cli.sync as sync_module
from hud.cli.app import CliError, app
from hud.cli.sync import _write_csv
from hud.eval import Task
from hud.utils.exceptions import HudRequestError

if TYPE_CHECKING:
    from pathlib import Path


def test_write_csv_flattens_args(tmp_path: Path) -> None:
    rows = [
        Task(env="e", id="solve", args={"n": 1}, slug="one"),
        Task(env="e", id="solve", args={"n": {"x": 2}}, slug="two"),
    ]
    rows = [row.model_dump() for row in rows]

    out = tmp_path / "tasks.csv"
    _write_csv(out, rows)

    csv_text = out.read_text()
    assert "slug,id,env,arg:n" in csv_text
    assert "one,solve,e,1" in csv_text
    assert 'two,solve,e,"{""x"": 2}"' in csv_text


def test_sync_env_noninteractive_requires_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sync_module.PlatformClient, "from_settings", lambda: object())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    with pytest.raises(CliError) as exc_info:
        sync_module.sync_env_command(name=None, directory=str(tmp_path), yes=False)

    assert exc_info.value.exit_code == 2


@pytest.mark.parametrize("failure", ["source", "export", "upload", "registry"])
def test_sync_failures_render_one_structured_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.default_project", None)
    source = tmp_path / "tasks.py"
    source.write_text("from hud.eval import Task\ntasks = [Task(env='e', id='solve')]\n")
    registry_id = "33333333-3333-4333-8333-333333333333"

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if url.endswith("/auth/me"):
            return {
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": "22222222-2222-4222-8222-222222222222",
            }
        if failure == "upload" and "/by-name/" in url:
            raise HudRequestError("missing", status_code=404)
        raise HudRequestError("Access denied", status_code=403)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    args = {
        "source": ["tasks", "demo", str(tmp_path / "missing.json")],
        "export": ["tasks", "demo", "--export", str(tmp_path / "out.json")],
        "upload": ["tasks", "demo", str(source), "--yes"],
        "registry": ["env", registry_id],
    }[failure]
    result = CliRunner().invoke(app, ["sync", *args, "--json"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["error"] == ("not_found" if failure == "source" else "permission_denied")
    assert payload["message"] not in (result.stderr or "")


def test_relinking_same_environment_reports_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    registry_id = "33333333-3333-4333-8333-333333333333"

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if url.endswith("/auth/me"):
            return {
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": "22222222-2222-4222-8222-222222222222",
            }
        assert url.endswith(f"/registry/{registry_id}")
        return {"id": registry_id, "name": "example"}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    args = ["sync", "env", registry_id, str(tmp_path), "--json"]
    for changed in (True, False):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["changed"] is changed
