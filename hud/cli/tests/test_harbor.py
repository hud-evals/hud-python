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
_RUN = "66666666-6666-4666-8666-666666666666"
_REPORT_FILE = "77777777-7777-4777-8777-777777777777"
_UPLOAD_URL = "https://storage.test/upload?signature=secret"
_REPORT_URL = "https://storage.test/report?signature=secret"


def _report(**changes: Any) -> dict[str, Any]:
    return {
        "schema_version": "harbor_import_report.v4",
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
        self.offers_harbor_import = True
        self.max_input_bytes = 1024**3
        self.run_pipeline_key = "harbor_import"
        self.run_statuses = ["completed"]
        self.report: Any = _report()
        self.storage_status = 200
        self.calls: list[tuple[str, str, Any]] = []
        self.run_queries: list[dict[str, list[str]]] = []
        self.uploaded: bytes | None = None
        self.upload_headers: httpx.Headers | None = None

    def body(self, method: str, path: str) -> Any:
        [body] = [body for called, at, body in self.calls if (called, at) == (method, path)]
        return body

    def paths(self, method: str) -> list[str]:
        return [path for called, path, _ in self.calls if called == method]

    def run(self, status: str) -> dict[str, Any]:
        finished = status != "queued"
        return {
            "id": _RUN,
            "pipeline_key": self.run_pipeline_key,
            "input_file_id": _FILE,
            "input_filename": "tb2.zip",
            "input_size_bytes": 10,
            "args": {"taskset_id": _TASKSET},
            "status": status,
            "phase": status if finished else "running",
            "result": {"summary": "Imported 1 task(s) and 1 run(s)"} if finished else None,
            "error": None,
            "outputs": [{"name": "report", "file_id": _REPORT_FILE, "filename": "r.json"}]
            if finished
            else [],
            "created_at": "2026-10-07T12:00:00Z",
        }

    def current_run(self) -> dict[str, Any]:
        # Each read moves to the next status; the last one stays.
        status = self.run_statuses.pop(0) if len(self.run_statuses) > 1 else self.run_statuses[0]
        return self.run(status)

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        parts = urlsplit(url)
        path = parts.path.removeprefix("/v2")
        self.calls.append((method, path, kwargs.get("json")))
        if path == "/auth/me":
            return {"user_id": _USER, "team_id": _TEAM}
        if path == "/data-pipelines":
            harbor = {
                "key": "harbor_import",
                "title": "Harbor import",
                "max_input_bytes": self.max_input_bytes,
            }
            return {"pipelines": [harbor] if self.offers_harbor_import else []}
        if path.startswith("/tasksets/by-name/"):
            name = path.removeprefix("/tasksets/by-name/")
            for taskset in self.tasksets.values():
                if taskset["name"] == name:
                    return {"taskset_id": taskset["id"], "name": name}
            raise HudRequestError("Taskset not found", status_code=404)
        if method == "GET" and path.startswith("/tasksets/"):
            return self.tasksets[path.removeprefix("/tasksets/")]
        if method == "POST" and path == "/tasksets":
            created = {"id": _TASKSET, "name": kwargs["json"]["name"], "can_edit": True}
            self.tasksets[_TASKSET] = created
            return created
        if method == "POST" and path == "/data":
            filename = kwargs["json"]["filename"]
            return {"file": {"id": _FILE, "filename": filename}, "upload_url": _UPLOAD_URL}
        if method == "POST" and path == f"/data/{_FILE}/complete":
            return {"id": _FILE, "filename": "tb2.zip", "size_bytes": 10}
        if method == "DELETE" and path == f"/data/{_FILE}":
            return None
        if path == f"/data/{_REPORT_FILE}/download":
            return {"download_url": _REPORT_URL}
        if method == "POST" and path == "/data-pipelines/runs":
            return self.current_run()
        if path == "/data-pipelines/runs":
            self.run_queries.append(parse_qs(parts.query))
            return {"runs": [self.current_run()], "total": 1}
        if path == f"/data-pipelines/runs/{_RUN}":
            return self.current_run()
        if path == f"/data-pipelines/runs/{_RUN}/cancel":
            return self.run("cancelled")
        raise AssertionError((method, url))

    def storage(self, request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            self.uploaded = request.read()
            self.upload_headers = request.headers
            return httpx.Response(self.storage_status)
        return httpx.Response(200, json=self.report)


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Platform:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.default_project", None)
    monkeypatch.setattr("hud.settings.settings.hud_web_url", "https://hud.test")
    monkeypatch.setattr("hud.data.time.sleep", lambda _seconds: None)
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


def test_import_creates_a_new_taskset_and_starts_the_run_into_it(
    platform: _Platform, folders: list[str]
) -> None:
    result, payload = _invoke("import", *folders, "--taskset", "TB2 Sample", "--yes")

    assert result.exit_code == 0, result.output
    assert platform.body("POST", "/tasksets") == {"name": "tb2-sample", "project_id": None}
    assert platform.body("POST", "/data-pipelines/runs") == {
        "pipeline_key": "harbor_import",
        "file_id": _FILE,
        "args": {"taskset_id": _TASKSET},
    }
    assert payload["taskset"] == {"id": _TASKSET, "name": "tb2-sample", "created": True}
    assert payload["run"]["status"] == "completed"
    assert payload["report"]["taskset_id"] == _TASKSET
    assert [trial["status"] for trial in payload["report"]["jobs"][0]["trials"]] == [
        "imported",
        "skipped",
    ]


def test_import_uploads_each_folder_under_its_name(platform: _Platform, folders: list[str]) -> None:
    result, _ = _invoke("import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == 0, result.output
    reservation = platform.body("POST", "/data")
    assert reservation["filename"] == "tb2.zip"
    assert reservation["content_type"] == "application/zip"
    assert reservation["pipeline_key"] == "harbor_import"
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
    assert "/data" not in platform.paths("POST")


def test_import_refuses_a_bundle_over_the_pipeline_limit(
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
    assert "/data-pipelines/runs" not in platform.paths("POST")
    assert "signature" not in result.output


def test_a_blocked_import_exits_with_failure_and_its_report(
    platform: _Platform, folders: list[str]
) -> None:
    platform.run_statuses = ["blocked"]
    platform.report = _report(error={"step": "add the tasks to the taskset", "message": "boom"})

    result, payload = _invoke("import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["run"]["status"] == "blocked"
    assert payload["report"]["error"] == {"step": "add the tasks to the taskset", "message": "boom"}


def test_no_wait_returns_the_started_run(platform: _Platform, folders: list[str]) -> None:
    platform.run_statuses = ["queued"]

    result, payload = _invoke("import", *folders, "--taskset", "tb2", "--yes", "--no-wait")

    assert result.exit_code == 0, result.output
    assert payload["run"]["phase"] == "running"
    assert payload["report"] is None
    assert f"/data-pipelines/runs/{_RUN}" not in platform.paths("GET")


def test_import_uploads_a_given_zip_as_it_is(platform: _Platform, tmp_path: Path) -> None:
    bundle = tmp_path / "prepared.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("hello/task.toml", "version = '1.0'\n")

    result, payload = _invoke("import", str(bundle), "--yes")

    assert result.exit_code == 0, result.output
    assert platform.uploaded == bundle.read_bytes()
    assert payload["taskset"]["name"] == "prepared"


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


def test_import_prints_the_report_tables(platform: _Platform, folders: list[str]) -> None:
    result = CliRunner().invoke(app, ["harbor", "import", *folders, "--taskset", "tb2", "--yes"])

    assert result.exit_code == 0, result.output
    assert "1 imported, 1 skipped" in result.output
    assert "it ran a task that is neither in the bundle nor in HUD" in result.output
    assert f"https://hud.test/tasksets/{_TASKSET}/imports" in result.output


def test_report_wait_follows_the_run_until_it_ends(platform: _Platform) -> None:
    platform.run_statuses = ["queued", "queued", "completed"]

    result, payload = _invoke("report", _RUN, "--wait")

    assert result.exit_code == 0, result.output
    assert payload["run"]["status"] == "completed"
    assert payload["report"]["taskset_id"] == _TASKSET


def test_report_of_a_running_import_without_wait_has_no_report(platform: _Platform) -> None:
    platform.run_statuses = ["queued"]

    result, payload = _invoke("report", _RUN)

    assert result.exit_code == 0, result.output
    assert payload["report"] is None
    assert f"/data/{_REPORT_FILE}/download" not in platform.paths("GET")


def test_report_of_an_unknown_schema_falls_back_to_the_summary(platform: _Platform) -> None:
    platform.report = {"schema_version": "harbor_import_report.v9"}

    result, payload = _invoke("report", _RUN)

    assert result.exit_code == 0, result.output
    assert payload["report"] is None
    assert payload["run"]["result"]["summary"] == "Imported 1 task(s) and 1 run(s)"


def test_report_refuses_a_run_of_another_pipeline(platform: _Platform) -> None:
    platform.run_pipeline_key = "pii_clean"

    result, payload = _invoke("report", _RUN)

    assert result.exit_code == ExitCode.USAGE
    assert payload["message"] == f"Run {_RUN} is a pii_clean run, not a Harbor import."


def test_cancel_requests_cancellation_of_a_running_import(platform: _Platform) -> None:
    platform.run_statuses = ["queued"]

    result, payload = _invoke("cancel", _RUN, "--yes")

    assert result.exit_code == 0, result.output
    assert f"/data-pipelines/runs/{_RUN}/cancel" in platform.paths("POST")
    assert payload["status"] == "cancelled"


def test_cancel_leaves_an_ended_import_alone(platform: _Platform) -> None:
    result, payload = _invoke("cancel", _RUN, "--yes")

    assert result.exit_code == 0, result.output
    assert platform.paths("POST") == []
    assert payload["status"] == "completed"


def test_imports_filters_by_taskset(platform: _Platform) -> None:
    platform.tasksets[_TASKSET] = {"id": _TASKSET, "name": "tb2", "can_edit": True}

    result, payload = _invoke("imports", "--taskset", "tb2")

    assert result.exit_code == 0, result.output
    assert [run["id"] for run in payload["runs"]] == [_RUN]
    assert platform.run_queries == [
        {"pipeline_key": ["harbor_import"], "limit": ["20"], "arg": [f"taskset_id:{_TASKSET}"]}
    ]
