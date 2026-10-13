"""Tests for ``hud harbor``."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpx
import pytest
from typer.testing import CliRunner, Result

from hud.cli import AuthScope, DirectoryState, ExitCode
from hud.cli.__main__ import app
from hud.utils.exceptions import HudRequestError
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from pathlib import Path

_USER = "11111111-1111-4111-8111-111111111111"
_TEAM = "22222222-2222-4222-8222-222222222222"
_TASKSET = "44444444-4444-4444-8444-444444444444"
_FILE = "55555555-5555-4555-8555-555555555555"
_IMPORT = "66666666-6666-4666-8666-666666666666"
_UPLOAD_URL = "https://storage.test/upload?signature=secret"
_LIVE = {"queued", "starting", "running", "cancelling"}


def _report(**changes: Any) -> dict[str, Any]:
    return {
        "schema_version": "harbor_import_report.v5",
        "taskset_id": _TASKSET,
        "tasks": [
            {
                "name": "hello",
                "source": "bundle",
                "path": "tasks/hello",
                "build_version": 1,
                "status": "imported",
            },
        ],
        "jobs": [
            {
                "path": "run-1",
                "name": "run-1",
                "job_id": "job-hud-1",
                "status": "imported",
                "trials": [
                    {"path": "run-1/hello__a", "status": "imported"},
                    {
                        "path": "run-1/gone__b",
                        "status": "skipped",
                        "reason": "it ran a task that is neither in the bundle nor in HUD",
                    },
                ],
            },
        ],
        "warnings": [],
        "error": None,
        **changes,
    }


class _Platform:
    """The platform and storage endpoints a Harbor import touches."""

    def __init__(self) -> None:
        self.tasksets: dict[str, dict[str, Any]] = {}
        self.has_imports_api = True
        self.offers_harbor_import = True
        self.max_input_bytes = 1024**3
        self.import_format = "harbor"
        self.import_statuses = ["completed"]
        self.import_error: str | None = None
        # None: the import published no report.
        self.report: Any = _report()
        self.storage_status = 200
        self.calls: list[tuple[str, str, Any]] = []
        self.list_queries: list[dict[str, list[str]]] = []
        self.uploaded: bytes | None = None
        self.upload_headers: httpx.Headers | None = None

    def body(self, method: str, path: str) -> Any:
        [body] = [body for called, at, body in self.calls if (called, at) == (method, path)]
        return body

    def paths(self, method: str) -> list[str]:
        return [path for called, path, _ in self.calls if called == method]

    def imported(self, status: str) -> dict[str, Any]:
        ended = status not in _LIVE
        reported = status in {"completed", "stopped"}
        return {
            "id": _IMPORT,
            "format": self.import_format,
            "taskset_id": _TASKSET,
            "file": {"id": _FILE, "filename": "tb2.zip", "size_bytes": 10},
            "status": status,
            "summary": "Imported 1 task(s) and 1 run(s)" if reported else None,
            "metrics": {"tasks_imported": 1.0, "trials_imported": 1.0} if reported else None,
            "error": self.import_error,
            "created_at": "2026-10-07T12:00:00Z",
            "completed_at": "2026-10-07T12:01:00Z" if ended else None,
        }

    def current_import(self) -> dict[str, Any]:
        # Each read moves to the next status; the last one stays.
        statuses = self.import_statuses
        status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        return self.imported(status)

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        parts = urlsplit(url)
        path = parts.path.removeprefix("/v2")
        self.calls.append((method, path, kwargs.get("json")))
        if path == "/auth/me":
            return {"user_id": _USER, "team_id": _TEAM}
        if path.startswith("/imports") and not self.has_imports_api:
            raise HudRequestError("Not Found", status_code=404)
        if path == "/imports/formats":
            harbor = {
                "format": "harbor",
                "title": "Harbor",
                "max_input_bytes": self.max_input_bytes,
            }
            return {"formats": [harbor] if self.offers_harbor_import else []}
        if path.startswith("/tasksets/by-name/"):
            name = path.removeprefix("/tasksets/by-name/")
            for taskset in self.tasksets.values():
                if taskset["name"] == name:
                    return {"taskset_id": taskset["id"], "name": name}
            raise HudRequestError("Taskset not found", status_code=404)
        if method == "GET" and path.startswith("/tasksets/"):
            taskset_id = path.removeprefix("/tasksets/")
            if taskset_id not in self.tasksets:
                raise HudRequestError("Taskset not found", status_code=404)
            return self.tasksets[taskset_id]
        if method == "POST" and path == "/tasksets":
            created = {"id": _TASKSET, "name": kwargs["json"]["name"], "can_edit": True}
            self.tasksets[_TASKSET] = created
            return created
        if method == "POST" and path == "/imports/uploads":
            filename = kwargs["json"]["filename"]
            return {"file": {"id": _FILE, "filename": filename}, "upload_url": _UPLOAD_URL}
        if method == "POST" and path == f"/data/{_FILE}/complete":
            return {"id": _FILE, "filename": "tb2.zip", "size_bytes": 10}
        if method == "DELETE" and path == f"/data/{_FILE}":
            return None
        if method == "POST" and path == "/imports":
            return self.current_import()
        if path == "/imports":
            self.list_queries.append(parse_qs(parts.query))
            return {"imports": [self.current_import()], "total": 1}
        if path == f"/imports/{_IMPORT}":
            return self.current_import()
        if path == f"/imports/{_IMPORT}/report":
            if self.report is None:
                raise HudRequestError("Import report not found", status_code=404)
            return self.report
        if path == f"/imports/{_IMPORT}/cancel":
            return self.imported("cancelling")
        if path.startswith("/imports/"):
            raise HudRequestError("Import not found", status_code=404)
        raise AssertionError((method, url))

    def storage(self, request: httpx.Request) -> httpx.Response:
        assert request.method == "PUT"
        self.uploaded = request.read()
        self.upload_headers = request.headers
        return httpx.Response(self.storage_status)


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Platform:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.default_project", None)
    monkeypatch.setattr("hud.settings.settings.hud_web_url", "https://hud.test")
    monkeypatch.setattr("hud.imports.time.sleep", lambda _seconds: None)
    fake = _Platform()
    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake.request)
    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(fake.storage), **kwargs),
    )
    return fake


@pytest.fixture
def folders(tmp_path: Path) -> list[str]:
    task = tmp_path / "tasks" / "hello"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text("version = '1.0'\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")
    job = tmp_path / "run-1"
    (job / "hello__a").mkdir(parents=True)
    for folder in (job, job / "hello__a"):
        (folder / "config.json").write_text("{}")
        (folder / "result.json").write_text("{}")
    return [str(tmp_path / "tasks"), str(job)]


def _invoke(*args: str) -> tuple[Result, dict[str, Any]]:
    result = CliRunner().invoke(app, ["harbor", *args, "--json"])
    return result, json.loads(result.stdout)


def test_import_creates_a_new_taskset_and_imports_into_it(
    platform: _Platform, folders: list[str]
) -> None:
    result, payload = _invoke("import", *folders, "--taskset", "TB2 Sample", "--yes")

    assert result.exit_code == 0, result.output
    assert platform.body("POST", "/tasksets") == {"name": "tb2-sample", "project_id": None}
    assert platform.body("POST", "/imports") == {
        "format": "harbor",
        "file_id": _FILE,
        "taskset_id": _TASKSET,
    }
    assert platform.paths("POST")[-2:] == [f"/data/{_FILE}/complete", "/imports"]
    assert payload["taskset"] == {"id": _TASKSET, "name": "tb2-sample", "created": True}
    assert payload["import"]["status"] == "completed"
    assert payload["import"]["summary"] == "Imported 1 task(s) and 1 run(s)"
    assert payload["report"]["taskset_id"] == _TASKSET
    assert [trial["status"] for trial in payload["report"]["jobs"][0]["trials"]] == [
        "imported",
        "skipped",
    ]


def test_import_uploads_each_folder_under_its_name(platform: _Platform, folders: list[str]) -> None:
    result, _ = _invoke("import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == 0, result.output
    reservation = platform.body("POST", "/imports/uploads")
    assert reservation["format"] == "harbor"
    assert reservation["filename"] == "tb2.zip"
    assert reservation["content_type"] == "application/zip"
    assert platform.upload_headers is not None
    assert platform.upload_headers["content-length"] == str(reservation["size_bytes"])
    assert platform.uploaded is not None
    with zipfile.ZipFile(io.BytesIO(platform.uploaded)) as bundle:
        assert sorted(bundle.namelist()) == [
            "run-1/config.json",
            "run-1/hello__a/config.json",
            "run-1/hello__a/result.json",
            "run-1/result.json",
            "tasks/hello/task.toml",
            "tasks/hello/tests/test.sh",
        ]


def test_import_defaults_to_the_taskset_named_after_the_first_folder(
    platform: _Platform, folders: list[str]
) -> None:
    platform.tasksets[_TASKSET] = {"id": _TASKSET, "name": "tasks", "can_edit": True}

    result, payload = _invoke("import", *folders, "--yes")

    assert result.exit_code == 0, result.output
    assert "/tasksets" not in platform.paths("POST")
    assert payload["taskset"] == {"id": _TASKSET, "name": "tasks", "created": False}


def test_rerunning_an_import_finds_the_taskset_its_name_created(
    platform: _Platform, folders: list[str]
) -> None:
    first, _ = _invoke("import", *folders, "--taskset", "TB2/Sample", "--yes")
    second, payload = _invoke("import", *folders, "--taskset", "TB2/Sample", "--yes")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert platform.paths("POST").count("/tasksets") == 1
    assert platform.paths("GET").count("/tasksets/by-name/tb2sample") == 2
    assert payload["taskset"] == {"id": _TASKSET, "name": "tb2sample", "created": False}


def test_import_refuses_an_unknown_taskset_id_instead_of_creating_one(
    platform: _Platform, folders: list[str]
) -> None:
    result, payload = _invoke("import", *folders, "--taskset", _TASKSET, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert payload["message"] == f"Taskset {_TASKSET} is inaccessible or deleted."
    assert platform.paths("POST") == []


def test_import_link_saves_the_taskset_as_the_directory_default(
    platform: _Platform, folders: list[str], tmp_path: Path
) -> None:
    result, _ = _invoke("import", *folders, "--taskset", "tb2", "--yes", "--link")

    assert result.exit_code == 0, result.output
    state = DirectoryState(AuthScope.resolve(PlatformClient.from_settings()), tmp_path)
    assert state.load().taskset_id == UUID(_TASKSET)


def test_import_refuses_a_taskset_the_caller_cannot_edit_before_uploading(
    platform: _Platform, folders: list[str]
) -> None:
    platform.tasksets[_TASKSET] = {"id": _TASKSET, "name": "theirs", "can_edit": False}

    result, payload = _invoke("import", *folders, "--taskset", _TASKSET, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "permission_denied"
    assert "/imports/uploads" not in platform.paths("POST")


def test_import_refuses_a_bundle_over_the_format_limit(
    platform: _Platform, folders: list[str]
) -> None:
    platform.max_input_bytes = 10

    result, payload = _invoke("import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.USAGE
    assert payload["message"].endswith("a Harbor import takes at most 10 bytes.")
    assert platform.paths("POST") == []


def test_dry_run_plans_without_writing(platform: _Platform, folders: list[str]) -> None:
    result, payload = _invoke("import", *folders, "--taskset", "tb2", "--dry-run")

    assert result.exit_code == 0, result.output
    assert payload["dry_run"] is True
    assert payload["taskset"] == {"id": None, "name": "tb2", "created": True}
    assert payload["bundle"]["filename"] == "tb2.zip"
    assert payload["bundle"]["files"] == 6
    assert platform.paths("POST") == []


def test_a_failed_upload_deletes_its_reservation_without_leaking_the_signed_url(
    platform: _Platform, folders: list[str]
) -> None:
    platform.storage_status = 403

    result, payload = _invoke("import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["message"] == "Storage refused the upload: HTTP 403"
    assert platform.paths("DELETE") == [f"/data/{_FILE}"]
    assert "/imports" not in platform.paths("POST")
    assert "signature" not in result.output


def test_a_stopped_import_exits_with_failure_and_its_report(
    platform: _Platform, folders: list[str]
) -> None:
    platform.import_statuses = ["stopped"]
    platform.report = _report(error={"step": "add the tasks to the taskset", "message": "boom"})

    result, payload = _invoke("import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["import"]["status"] == "stopped"
    assert payload["report"]["error"] == {"step": "add the tasks to the taskset", "message": "boom"}


def test_a_failed_import_without_a_report_exits_with_its_error(
    platform: _Platform, folders: list[str]
) -> None:
    platform.import_statuses = ["failed"]
    platform.import_error = "the importer crashed"

    result = CliRunner().invoke(app, ["harbor", "import", *folders, "--taskset", "tb2", "--yes"])

    assert result.exit_code == ExitCode.FAILURE
    assert "The import failed: the importer crashed" in result.output
    assert f"/imports/{_IMPORT}/report" not in platform.paths("GET")


def test_a_stopped_import_without_a_report_shows_its_summary(
    platform: _Platform, folders: list[str]
) -> None:
    platform.import_statuses = ["stopped"]
    platform.report = None

    result, payload = _invoke("import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["report"] is None
    assert payload["import"]["summary"] == "Imported 1 task(s) and 1 run(s)"


def test_no_wait_returns_the_started_import(platform: _Platform, folders: list[str]) -> None:
    platform.import_statuses = ["queued"]

    result, payload = _invoke("import", *folders, "--taskset", "tb2", "--yes", "--no-wait")

    assert result.exit_code == 0, result.output
    assert payload["import"]["status"] == "queued"
    assert payload["report"] is None
    assert f"/imports/{_IMPORT}" not in platform.paths("GET")


def test_import_uploads_a_given_zip_as_it_is(platform: _Platform, tmp_path: Path) -> None:
    bundle = tmp_path / "prepared.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("hello/task.toml", "version = '1.0'\n")

    result, payload = _invoke("import", str(bundle), "--yes")

    assert result.exit_code == 0, result.output
    assert platform.uploaded == bundle.read_bytes()
    assert payload["taskset"]["name"] == "prepared"


def test_import_packs_a_folder_named_like_a_zip(platform: _Platform, tmp_path: Path) -> None:
    folder = tmp_path / "release.zip"
    folder.mkdir()
    (folder / "task.toml").write_text("version = '1.0'\n")

    result, payload = _invoke("import", str(folder), "--taskset", "tb2", "--yes")

    assert result.exit_code == 0, result.output
    assert payload["bundle"]["files"] == 1
    assert platform.uploaded is not None
    with zipfile.ZipFile(io.BytesIO(platform.uploaded)) as bundle:
        assert bundle.namelist() == ["release.zip/task.toml"]


def test_import_refuses_a_zip_mixed_with_folders_before_any_request(
    platform: _Platform, folders: list[str], tmp_path: Path
) -> None:
    bundle = tmp_path / "prepared.zip"
    bundle.write_bytes(b"zip")

    result, payload = _invoke("import", str(bundle), folders[0], "--yes")

    assert result.exit_code == ExitCode.USAGE
    assert payload["message"] == "Pass either one .zip bundle or folders, not both or several zips."
    assert platform.calls == []


def test_import_fails_where_the_platform_offers_no_harbor_import(
    platform: _Platform, folders: list[str]
) -> None:
    platform.offers_harbor_import = False

    result, payload = _invoke("import", *folders, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert payload["message"].startswith("Harbor import is not available on ")


def test_import_fails_where_the_platform_has_no_imports(
    platform: _Platform, folders: list[str]
) -> None:
    platform.has_imports_api = False

    result, payload = _invoke("import", *folders, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert payload["message"].startswith("Harbor import is not available on ")


def test_import_prints_the_report_tables(platform: _Platform, folders: list[str]) -> None:
    result = CliRunner().invoke(app, ["harbor", "import", *folders, "--taskset", "tb2", "--yes"])

    assert result.exit_code == 0, result.output
    assert "1 imported, 1 skipped" in result.output
    assert "it ran a task that is neither in the bundle nor in HUD" in result.output
    assert f"https://hud.test/tasksets/{_TASKSET}/imports" in result.output


def test_report_wait_follows_the_import_until_it_ends(platform: _Platform) -> None:
    platform.import_statuses = ["queued", "running", "completed"]

    result, payload = _invoke("report", _IMPORT, "--wait")

    assert result.exit_code == 0, result.output
    assert payload["import"]["status"] == "completed"
    assert payload["report"]["taskset_id"] == _TASKSET


def test_report_of_a_running_import_without_wait_has_no_report(platform: _Platform) -> None:
    platform.import_statuses = ["running"]

    result, payload = _invoke("report", _IMPORT)

    assert result.exit_code == 0, result.output
    assert payload["report"] is None
    assert f"/imports/{_IMPORT}/report" not in platform.paths("GET")


def test_report_shows_tasks_an_import_updated(platform: _Platform) -> None:
    [task] = platform.report["tasks"]
    platform.report["tasks"] = [{**task, "build_version": 2, "status": "updated"}]

    result = CliRunner().invoke(app, ["harbor", "report", _IMPORT])

    assert result.exit_code == 0, result.output
    assert "updated" in result.output


def test_report_reads_a_v4_report(platform: _Platform) -> None:
    platform.report = _report(schema_version="harbor_import_report.v4")

    result, payload = _invoke("report", _IMPORT)

    assert result.exit_code == 0, result.output
    assert payload["report"]["schema_version"] == "harbor_import_report.v4"
    assert payload["report"]["tasks"][0]["status"] == "imported"


def test_report_of_an_unknown_schema_falls_back_to_the_summary(platform: _Platform) -> None:
    platform.report = {"schema_version": "harbor_import_report.v9"}

    result, payload = _invoke("report", _IMPORT)

    assert result.exit_code == 0, result.output
    assert payload["report"] is None
    assert payload["import"]["summary"] == "Imported 1 task(s) and 1 run(s)"


def test_report_refuses_an_import_of_another_format(platform: _Platform) -> None:
    platform.import_format = "parquet"

    result, payload = _invoke("report", _IMPORT)

    assert result.exit_code == ExitCode.USAGE
    assert payload["message"] == f"Import {_IMPORT} is a parquet import, not a Harbor import."


def test_report_of_an_unknown_import_is_not_found(platform: _Platform) -> None:
    result, payload = _invoke("report", "88888888-8888-4888-8888-888888888888")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"


def test_cancel_requests_cancellation_of_a_running_import(platform: _Platform) -> None:
    platform.import_statuses = ["running"]

    result, payload = _invoke("cancel", _IMPORT, "--yes")

    assert result.exit_code == 0, result.output
    assert f"/imports/{_IMPORT}/cancel" in platform.paths("POST")
    assert payload["status"] == "cancelling"


def test_cancel_leaves_an_ended_import_alone(platform: _Platform) -> None:
    result, payload = _invoke("cancel", _IMPORT, "--yes")

    assert result.exit_code == 0, result.output
    assert platform.paths("POST") == []
    assert payload["status"] == "completed"


def test_imports_filters_by_taskset(platform: _Platform) -> None:
    platform.tasksets[_TASKSET] = {"id": _TASKSET, "name": "tb2", "can_edit": True}

    result, payload = _invoke("imports", "--taskset", "tb2")

    assert result.exit_code == 0, result.output
    assert [item["id"] for item in payload["imports"]] == [_IMPORT]
    assert platform.list_queries == [
        {"limit": ["20"], "format": ["harbor"], "taskset_id": [_TASKSET]}
    ]


def test_imports_refuses_an_unknown_taskset_id(platform: _Platform) -> None:
    result, payload = _invoke("imports", "--taskset", _TASKSET)

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert platform.list_queries == []


def test_imports_refuses_a_limit_the_platform_does_not_page(platform: _Platform) -> None:
    result = CliRunner().invoke(app, ["harbor", "imports", "--limit", "101"])

    assert result.exit_code == ExitCode.USAGE
    assert platform.calls == []
