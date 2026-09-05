"""Project placement behavior for ``hud sync tasks``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import hud.cli.sync as sync_module
from hud.eval import Task, Taskset

if TYPE_CHECKING:
    from pathlib import Path


class _ReadOnlyPlatform:
    def get(self, url: str) -> dict[str, Any]:
        assert url == "/projects"
        return {
            "projects": [
                {
                    "id": "33333333-3333-4333-8333-333333333333",
                    "name": "locked-down",
                    "capabilities": {"create": False},
                }
            ]
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
    monkeypatch.setattr(sync_module, "require_api_key", lambda _: None)
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
        project="locked-down",
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
