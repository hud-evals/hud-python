"""Tests for ``hud.cli.sync``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from hud.cli import AuthScope, CliError, DirectoryState
from hud.cli.__main__ import app
from hud.cli.sync import RegistryEnvironment
from hud.eval import Task, Taskset
from hud.utils.exceptions import HudRequestError
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path


_TASKSET_ID = "44444444-4444-4444-8444-444444444444"
_READONLY_PROJECT = "33333333-3333-4333-8333-333333333333"
_WRITABLE_PROJECT = "22222222-2222-4222-8222-222222222222"
_ROW = {"env": "example", "id": "solve", "slug": "one"}


def _stub_platform(
    monkeypatch: pytest.MonkeyPatch, *, remote_rows: list[dict[str, Any]], uploads: list[Any]
) -> None:
    """A platform holding taskset ``demo`` with ``remote_rows`` and two projects."""

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if url.endswith("/auth/me"):
            return {
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": _WRITABLE_PROJECT,
            }
        if url.endswith(f"/projects/{_READONLY_PROJECT}"):
            return {"id": _READONLY_PROJECT, "name": "locked-down", "capabilities": {}}
        if url.endswith(f"/projects/{_WRITABLE_PROJECT}"):
            return {
                "id": _WRITABLE_PROJECT,
                "name": "browser-evals",
                "capabilities": {"create": True},
            }
        if url.endswith("/tasksets/by-name/demo"):
            return {"taskset_id": _TASKSET_ID, "name": "demo"}
        if url.endswith(f"/tasksets/{_TASKSET_ID}"):
            return {"id": _TASKSET_ID, "name": "demo"}
        if url.endswith(f"/tasksets/{_TASKSET_ID}/export"):
            return {
                "name": "demo",
                "tasks": [
                    {"env": row["env"], "scenario": row["id"], "name": row["slug"]}
                    for row in remote_rows
                ],
            }
        if url.endswith("/tasks/upload"):
            uploads.append(kwargs["json"])
            return {"taskset_id": _TASKSET_ID, "tasks_created": 1}
        raise AssertionError((method, url))

    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.default_project", None)
    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)


@pytest.mark.parametrize("dry_run", [False, True])
def test_read_only_project_allows_no_op_and_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dry_run: bool
) -> None:
    """A read-only project is only refused when something would actually upload."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.json").write_text(json.dumps([_ROW]))
    uploads: list[Any] = []
    # Up to date: identical remote row; dry run: empty remote, so a create is planned.
    _stub_platform(monkeypatch, remote_rows=[] if dry_run else [_ROW], uploads=uploads)

    args = [
        "sync",
        "tasks",
        "demo",
        "tasks.json",
        "--project",
        _READONLY_PROJECT,
        "--yes",
        "--json",
    ]
    result = CliRunner().invoke(app, [*args, "--dry-run"] if dry_run else args)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["dry_run"] is dry_run
    assert uploads == []

    if dry_run:
        refused = CliRunner().invoke(app, args)
        assert refused.exit_code == 1, refused.output
        assert json.loads(refused.stdout)["error"] == "permission_denied"
        assert uploads == []


def test_project_override_does_not_pin_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks.json").write_text(json.dumps([_ROW]))
    uploads: list[Any] = []
    _stub_platform(monkeypatch, remote_rows=[], uploads=uploads)

    result = CliRunner().invoke(
        app,
        ["sync", "tasks", "demo", "tasks.json", "--project", _WRITABLE_PROJECT, "--yes", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert uploads[0]["project_id"] == _WRITABLE_PROJECT
    assert not (tmp_path / ".hud" / "config.json").exists()


@pytest.mark.parametrize("stale", [False, True])
def test_sync_uses_stored_id_after_rename_and_never_recreates_stale_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    source = tmp_path / "tasks.py"
    source.write_text(
        "from hud.eval import Task\ntasks = [Task(env='example', id='solve', slug='one')]\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.default_project", None)
    taskset_id = "55555555-5555-4555-8555-555555555555"
    uploads: list[dict[str, Any]] = []

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if url.endswith("/auth/me"):
            return {
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": "22222222-2222-4222-8222-222222222222",
            }
        if "/by-name/" in url:
            raise HudRequestError("missing", status_code=404)
        if url.endswith("/tasks/upload"):
            uploads.append(kwargs["json"])
            return {"taskset_id": kwargs["json"].get("taskset_id", taskset_id), "tasks_created": 1}
        if "/tasksets/" in url:
            if stale:
                raise HudRequestError("deleted", status_code=404)
            return {"id": taskset_id, "name": "renamed", "tasks": []}
        raise AssertionError(url)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    first = CliRunner().invoke(app, ["sync", "tasks", "demo", str(source), "--yes", "--json"])
    assert first.exit_code == 0, first.output
    state = DirectoryState(AuthScope.resolve(PlatformClient.from_settings()), tmp_path)
    before = state.load()
    second = CliRunner().invoke(app, ["sync", "tasks", "--yes", "--json", "--force"])
    assert state.load() == before
    if stale:
        assert second.exit_code == 1, second.output
        assert json.loads(second.stdout)["error"] == "not_found"
        assert len(uploads) == 1
    else:
        assert second.exit_code == 0, second.output
        assert uploads[-1]["taskset_id"] == taskset_id
        assert uploads[-1]["taskset_name"] == "renamed"

        other = "66666666-6666-4666-8666-666666666666"
        args = ["sync", "tasks", other, str(source), "--force", "--yes", "--json"]
        override = CliRunner().invoke(app, args)
        assert override.exit_code == 0, override.output
        assert uploads[-1]["taskset_id"] == other
        assert state.load() == before
        planned_link = CliRunner().invoke(app, [*args, "--link", "--dry-run"])
        assert planned_link.exit_code == 0, planned_link.output
        assert state.load() == before
        relinked = CliRunner().invoke(app, [*args, "--link"])
        assert relinked.exit_code == 0, relinked.output
        assert str(state.load().taskset_id) == other

        alias = CliRunner().invoke(app, ["sync", "tasks", "--yes", "--json", "--force"])
        assert alias.exit_code == 0, alias.output
        assert json.loads(alias.stdout)["taskset_id"] == other


@pytest.mark.parametrize(
    ("status_code", "exit_code", "error"),
    [
        (400, 1, "failure"),
        (403, 1, "permission_denied"),
        (500, 1, "server_error"),
    ],
)
def test_rejected_upload_exits_with_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status_code: int,
    exit_code: int,
    error: str,
) -> None:
    project_id = "22222222-2222-4222-8222-222222222222"
    detail = "Taskset belongs to another Project" if status_code == 400 else "Upload rejected"
    source = tmp_path / "tasks.json"
    source.write_text(json.dumps([{"env": "example", "id": "solve", "slug": "one"}]))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.default_project", None)

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        if url.endswith("/auth/me"):
            return {
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": "22222222-2222-4222-8222-222222222222",
            }
        if url.endswith(f"/projects/{project_id}"):
            return {"id": project_id, "name": "browser-evals", "capabilities": {"create": True}}
        if "/by-name/" in url:
            raise HudRequestError("missing", status_code=404)
        if url.endswith("/tasks/upload"):
            raise HudRequestError(detail, status_code=status_code, response_json={"detail": detail})
        raise AssertionError(url)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    result = CliRunner().invoke(
        app,
        [
            "sync",
            "tasks",
            "demo",
            str(source),
            "--project",
            project_id,
            "--force",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == exit_code, result.output
    payload = json.loads(result.stdout)
    assert payload["error"] == error
    assert detail in payload["message"]
    assert "Sync complete" not in result.output
    assert not (tmp_path / ".hud" / "config.json").exists()


def test_export_csv_flattens_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    remote = Taskset(
        "demo",
        [
            Task(env="e", id="solve", args={"n": 1}, slug="one"),
            Task(env="e", id="solve", args={"n": {"x": 2}}, slug="two"),
        ],
    )
    monkeypatch.setattr(Taskset, "from_api", classmethod(lambda cls, name: remote))
    monkeypatch.setattr(
        "hud.utils.platform.make_request_sync",
        lambda method, url, **kw: {
            "user_id": "11111111-1111-4111-8111-111111111111",
            "team_id": _WRITABLE_PROJECT,
        },
    )

    result = CliRunner().invoke(app, ["sync", "tasks", "demo", "--export", "tasks.csv", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "action": "export",
        "taskset": "demo",
        "path": "tasks.csv",
        "task_count": 2,
    }
    csv_text = (tmp_path / "tasks.csv").read_text()
    assert "slug,id,env,arg:n" in csv_text
    assert "one,solve,e,1" in csv_text
    assert 'two,solve,e,"{""x"": 2}"' in csv_text


def test_sync_env_noninteractive_requires_name(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["sync", "env", "--json"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error"] == "usage"


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


def test_resolve_verifies_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    def request(method: str, url: str, **kwargs: object) -> dict[str, str]:
        assert url.endswith("/registry/12345678-1234-5678-1234-567812345678")
        return {"id": "12345678-1234-5678-1234-567812345678", "name": "verified"}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    env = RegistryEnvironment.resolve(
        PlatformClient("https://api.example", "key"),
        "12345678-1234-5678-1234-567812345678",
    )

    assert env == RegistryEnvironment(
        id="12345678-1234-5678-1234-567812345678",
        name="verified",
    )


def test_get_registry_environment_treats_404_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, **kwargs: object) -> dict[str, Any]:
        raise HudRequestError("not found", status_code=404)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake_request)

    with pytest.raises(CliError, match="inaccessible or deleted") as error:
        RegistryEnvironment.resolve(
            PlatformClient("https://api.example", "key"), "12345678-1234-5678-1234-567812345678"
        )
    assert error.value.exit_code == 1


def test_resolve_matches_a_name_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {"id": "12345678-1234-5678-1234-567812345678", "name": "browser"},
        {"id": "87654321-4321-8765-4321-876543218765", "name": "browser-anchor"},
    ]

    def request(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        parts = urlsplit(url)
        assert parts.path.endswith("/registry")
        search = parse_qs(parts.query)["search"][0]
        items = [record for record in records if search in record["name"]]
        return {"items": items, "total": len(items)}

    monkeypatch.setattr("hud.utils.platform.make_request_sync", request)
    platform = PlatformClient("https://api.example", "key")

    assert RegistryEnvironment.resolve(platform, "Browser").id == records[0]["id"]
    assert RegistryEnvironment.resolve(platform, "browser_anchor").id == records[1]["id"]
    with pytest.raises(CliError, match="No environment named 'anchor'") as error:
        RegistryEnvironment.resolve(platform, "anchor")
    assert error.value.error == "not_found"
