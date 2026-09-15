"""Project placement behavior for ``hud sync tasks``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

import hud.cli.sync as sync_module
from hud.cli.app import AuthScope, DirectoryState, app
from hud.eval import Task, Taskset
from hud.utils.exceptions import HudRequestError
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path


class _ReadOnlyPlatform:
    api_url = "https://api.example"

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if url == "/auth/me":
            return {
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": "22222222-2222-4222-8222-222222222222",
            }
        assert url.startswith("/projects/")
        return {
            "id": "33333333-3333-4333-8333-333333333333",
            "name": "locked-down",
            "capabilities": {"create": False},
        }


class _WritablePlatform:
    api_url = "https://api.example"

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if url == "/auth/me":
            return {
                "user_id": "11111111-1111-4111-8111-111111111111",
                "team_id": "22222222-2222-4222-8222-222222222222",
            }
        assert url.startswith("/projects/")
        return {
            "id": "22222222-2222-4222-8222-222222222222",
            "name": "browser-evals",
            "capabilities": {"create": True},
        }


def _run_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remote: Taskset,
    *,
    dry_run: bool,
) -> None:
    task = Task(env="example", id="solve", slug="one")
    local = Taskset("demo", [task])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sync_module.PlatformClient,
        "from_settings",
        lambda: _ReadOnlyPlatform(),
    )
    monkeypatch.setattr(sync_module, "_load_local_taskset", lambda *args, **kwargs: local)
    monkeypatch.setattr(sync_module, "_fetch_remote_taskset", lambda *args, **kwargs: remote)
    monkeypatch.setattr(
        sync_module,
        "upload_taskset",
        lambda *args, **kwargs: pytest.fail("read-only no-op must not upload"),
    )

    sync_module.sync_tasks_command(
        taskset="demo",
        source=".",
        taskset_id=None,
        project="33333333-3333-4333-8333-333333333333",
        task_filter=None,
        exclude=None,
        yes=True,
        dry_run=dry_run,
        force=False,
        export=None,
    )


def test_read_only_project_allows_up_to_date_sync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = Task(env="example", id="solve", slug="one")
    _run_sync(monkeypatch, tmp_path, Taskset("demo", [task]), dry_run=False)


def test_read_only_project_allows_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _run_sync(monkeypatch, tmp_path, Taskset("demo", []), dry_run=True)


def test_project_override_does_not_pin_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = Task(env="example", id="solve", slug="one")
    local = Taskset("demo", [task])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sync_module.PlatformClient,
        "from_settings",
        lambda: _WritablePlatform(),
    )
    monkeypatch.setattr(sync_module, "_load_local_taskset", lambda *args, **kwargs: local)
    monkeypatch.setattr(
        sync_module,
        "_fetch_remote_taskset",
        lambda *args, **kwargs: Taskset("demo", []),
    )

    def upload(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["project_id"] == "22222222-2222-4222-8222-222222222222"
        return {"taskset_id": "taskset-1", "tasks_created": 1}

    monkeypatch.setattr(sync_module, "upload_taskset", upload)

    sync_module.sync_tasks_command(
        taskset="demo",
        source=".",
        taskset_id=None,
        project="22222222-2222-4222-8222-222222222222",
        task_filter=None,
        exclude=None,
        yes=True,
        dry_run=False,
        force=False,
        export=None,
    )

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
