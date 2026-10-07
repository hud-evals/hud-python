"""Tests for ``hud harbor``."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from typer.testing import CliRunner

from hud.cli import ExitCode
from hud.cli.__main__ import app
from hud.utils.exceptions import HudRequestError

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

runner = CliRunner()


def _report(**changes: Any) -> dict[str, Any]:
    return {
        "schema_version": "harbor_import_report.v4",
        "taskset_id": _TASKSET,
        "source": {"id": _FILE, "filename": "tb2.zip"},
        "tasks": [
            {
                "name": "hello",
                "source": "bundle",
                "path": "tasks/hello",
                "digest": "sha256:" + "a" * 64,
                "environment_id": "env-1",
                "build_id": "build-1",
                "build_version": 1,
                "task_id": "task-1",
                "status": "imported",
                "reason": None,
            },
        ],
        "jobs": [
            {
                "path": "jobs/run-1",
                "external_id": "job-1",
                "name": "run-1",
                "job_id": "job-hud-1",
                "job_status": "completed",
                "status": "imported",
                "reason": None,
                "trials": [
                    {"path": "jobs/run-1/hello__a", "status": "imported", "trace_id": "t-1"},
                    {
                        "path": "jobs/run-1/gone__b",
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

    def __init__(
        self,
        *,
        tasksets: dict[str, dict[str, Any]] | None = None,
        run_status: str = "completed",
        report: Any = None,
        storage_status: int = 200,
        max_input_bytes: int = 1024**3,
    ) -> None:
        self.tasksets = tasksets or {}
        self.run_status = run_status
        self.report = _report() if report is None else report
        self.storage_status = storage_status
        self.max_input_bytes = max_input_bytes
        self.calls: list[tuple[str, str, Any]] = []
        self.run_queries: list[dict[str, list[str]]] = []
        self.uploaded: bytes | None = None
        self.upload_headers: httpx.Headers | None = None

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        finished = self.run_status != "queued"
        return {
            "id": _RUN,
            "pipeline_key": "harbor_import",
            "input_file_id": _FILE,
            "input_filename": "tb2.zip",
            "input_size_bytes": 10,
            "args": args,
            "status": self.run_status,
            "phase": self.run_status if finished else "running",
            "result": {"summary": "Imported 1 task(s) and 1 run(s) from 1 job(s)"}
            if finished
            else None,
            "error": None,
            "outputs": [{"name": "report", "file_id": _REPORT_FILE, "filename": "r.json"}]
            if finished
            else [],
            "created_at": "2026-10-07T12:00:00Z",
            "completed_at": "2026-10-07T12:01:00Z" if finished else None,
        }

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        parts = urlsplit(url)
        path = parts.path.removeprefix("/v2")
        self.calls.append((method, path, kwargs.get("json")))
        if path == "/auth/me":
            return {"user_id": _USER, "team_id": _TEAM}
        if path == "/data-pipelines":
            return {
                "pipelines": [
                    {"key": "pii_clean", "title": "PII", "max_input_bytes": 1},
                    {
                        "key": "harbor_import",
                        "title": "Harbor import",
                        "max_input_bytes": self.max_input_bytes,
                    },
                ],
            }
        if path.startswith("/tasksets/by-name/"):
            name = path.removeprefix("/tasksets/by-name/")
            match = next((t for t in self.tasksets.values() if t["name"] == name), None)
            if match is None:
                raise HudRequestError("Taskset not found", status_code=404)
            return {"taskset_id": match["id"], "name": name}
        if method == "GET" and path.startswith("/tasksets/"):
            return self.tasksets[path.removeprefix("/tasksets/")]
        if method == "POST" and path == "/tasksets":
            created = {"id": _TASKSET, "name": kwargs["json"]["name"], "can_edit": True}
            self.tasksets[_TASKSET] = created
            return created
        if method == "POST" and path == "/data":
            body = kwargs["json"]
            return {"file": {"id": _FILE, "filename": body["filename"]}, "upload_url": _UPLOAD_URL}
        if method == "POST" and path == f"/data/{_FILE}/complete":
            return {"id": _FILE, "filename": "tb2.zip", "size_bytes": 10}
        if method == "DELETE" and path == f"/data/{_FILE}":
            return None
        if method == "POST" and path == "/data-pipelines/runs":
            return self.run(kwargs["json"]["args"])
        if method == "GET" and path == f"/data-pipelines/runs/{_RUN}":
            return self.run({"taskset_id": _TASKSET})
        if method == "GET" and path == "/data-pipelines/runs":
            self.run_queries.append(parse_qs(parts.query))
            return {"runs": [self.run({"taskset_id": _TASKSET})], "total": 1}
        if method == "GET" and path == f"/data/{_REPORT_FILE}/download":
            return {"url": _REPORT_URL, "download_url": _REPORT_URL}
        raise AssertionError((method, url))

    def storage(self, request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            self.uploaded = request.read()
            self.upload_headers = request.headers
            return httpx.Response(self.storage_status)
        return httpx.Response(200, json=self.report)

    def paths(self, method: str) -> list[str]:
        return [path for called, path, _ in self.calls if called == method]


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Platform:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")
    monkeypatch.setattr("hud.settings.settings.default_project", None)
    monkeypatch.setattr("hud.settings.settings.hud_web_url", "https://hud.test")
    fake = _Platform()
    monkeypatch.setattr("hud.utils.platform.make_request_sync", fake.request)
    client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(fake.storage), **kwargs),
    )
    return fake


def _harbor_folders(root: Path) -> list[Path]:
    task = root / "tasks" / "hello"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text("version = '1.0'\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")
    job = root / "run-1"
    (job / "hello__a").mkdir(parents=True)
    for folder in (job, job / "hello__a"):
        (folder / "config.json").write_text("{}")
        (folder / "result.json").write_text("{}")
    return [root / "tasks", job]


def _invoke(*args: str) -> Any:
    result = runner.invoke(app, ["harbor", *args, "--json"])
    return result, json.loads(result.stdout)


def test_import_creates_the_taskset_uploads_the_bundle_and_reports(
    platform: _Platform, tmp_path: Path
) -> None:
    folders = _harbor_folders(tmp_path)

    result, payload = _invoke("import", *map(str, folders), "--taskset", "TB2 Sample", "--yes")

    assert result.exit_code == 0, result.output
    assert ("POST", "/tasksets", {"name": "tb2-sample", "project_id": None}) in platform.calls
    reservation = next(body for method, path, body in platform.calls if path == "/data")
    assert reservation["pipeline_key"] == "harbor_import"
    assert reservation["filename"] == "tb2-sample.zip"
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
    start = next(body for method, path, body in platform.calls if path == "/data-pipelines/runs")
    assert start == {
        "pipeline_key": "harbor_import",
        "file_id": _FILE,
        "args": {"taskset_id": _TASKSET},
    }
    assert payload["taskset"] == {"id": _TASKSET, "name": "tb2-sample", "created": True}
    assert payload["run"]["status"] == "completed"
    assert payload["report"]["jobs"][0]["trials"][1]["status"] == "skipped"


def test_import_defaults_to_a_taskset_named_after_the_first_folder(
    platform: _Platform, tmp_path: Path
) -> None:
    folders = _harbor_folders(tmp_path)
    platform.tasksets[_TASKSET] = {"id": _TASKSET, "name": "tasks", "can_edit": True}

    result, payload = _invoke("import", *map(str, folders), "--yes")

    assert result.exit_code == 0, result.output
    assert "/tasksets" not in platform.paths("POST")
    assert payload["taskset"] == {"id": _TASKSET, "name": "tasks", "created": False}


def test_import_refuses_a_taskset_the_caller_cannot_edit_before_uploading(
    platform: _Platform, tmp_path: Path
) -> None:
    folders = _harbor_folders(tmp_path)
    platform.tasksets[_TASKSET] = {"id": _TASKSET, "name": "theirs", "can_edit": False}

    result, payload = _invoke("import", *map(str, folders), "--taskset", _TASKSET, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "permission_denied"
    assert "/data" not in platform.paths("POST")


def test_import_refuses_a_bundle_over_the_pipeline_limit(
    platform: _Platform, tmp_path: Path
) -> None:
    folders = _harbor_folders(tmp_path)
    platform.max_input_bytes = 10

    result, payload = _invoke("import", *map(str, folders), "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.USAGE
    assert "at most" in payload["message"]
    assert "/data" not in platform.paths("POST")
    assert "/tasksets" not in platform.paths("POST")


def test_dry_run_makes_no_changes(platform: _Platform, tmp_path: Path) -> None:
    folders = _harbor_folders(tmp_path)

    result, payload = _invoke("import", *map(str, folders), "--taskset", "tb2", "--dry-run")

    assert result.exit_code == 0, result.output
    assert payload["dry_run"] is True
    assert payload["bundle"]["files"] == 6
    assert platform.paths("POST") == []


def test_a_failed_upload_deletes_its_reservation_without_leaking_the_signed_url(
    platform: _Platform, tmp_path: Path
) -> None:
    folders = _harbor_folders(tmp_path)
    platform.storage_status = 403

    result, payload = _invoke("import", *map(str, folders), "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["message"] == "Storage refused the upload: HTTP 403"
    assert f"/data/{_FILE}" in platform.paths("DELETE")
    assert "/data-pipelines/runs" not in platform.paths("POST")
    assert "signature" not in result.output


def test_a_blocked_import_fails_with_its_report(platform: _Platform, tmp_path: Path) -> None:
    folders = _harbor_folders(tmp_path)
    platform.run_status = "blocked"
    platform.report = _report(error={"step": "add the tasks to the taskset", "message": "boom"})

    result, payload = _invoke("import", *map(str, folders), "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["run"]["status"] == "blocked"
    assert payload["report"]["error"]["step"] == "add the tasks to the taskset"


def test_no_wait_returns_the_started_run(platform: _Platform, tmp_path: Path) -> None:
    folders = _harbor_folders(tmp_path)
    platform.run_status = "queued"

    result, payload = _invoke("import", *map(str, folders), "-t", "tb2", "--yes", "--no-wait")

    assert result.exit_code == 0, result.output
    assert payload["run"]["phase"] == "running"
    assert payload["report"] is None


def test_import_takes_one_zip_as_it_is(platform: _Platform, tmp_path: Path) -> None:
    bundle = tmp_path / "prepared.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("hello/task.toml", "version = '1.0'\n")

    result, payload = _invoke("import", str(bundle), "--yes")

    assert result.exit_code == 0, result.output
    assert platform.uploaded == bundle.read_bytes()
    assert payload["taskset"]["name"] == "prepared"


def test_import_refuses_a_zip_mixed_with_folders(platform: _Platform, tmp_path: Path) -> None:
    folders = _harbor_folders(tmp_path)
    bundle = tmp_path / "prepared.zip"
    bundle.write_bytes(b"zip")

    result, payload = _invoke("import", str(bundle), str(folders[0]), "--yes")

    assert result.exit_code == ExitCode.USAGE
    assert "one .zip bundle or folders" in payload["message"]
    assert platform.calls == []


def test_import_fails_where_the_platform_offers_no_harbor_import(
    platform: _Platform, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folders = _harbor_folders(tmp_path)
    original = platform.request

    def without_pipeline(method: str, url: str, **kwargs: Any) -> Any:
        if url.endswith("/data-pipelines"):
            return {"pipelines": []}
        return original(method, url, **kwargs)

    monkeypatch.setattr("hud.utils.platform.make_request_sync", without_pipeline)

    result, payload = _invoke("import", *map(str, folders), "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert "Harbor import is not available" in payload["message"]


def test_report_of_an_unknown_schema_shows_the_summary_only(platform: _Platform) -> None:
    platform.report = {"schema_version": "harbor_import_report.v9"}

    result, payload = _invoke("report", _RUN)

    assert result.exit_code == 0, result.output
    assert payload["report"] is None
    assert payload["run"]["result"]["summary"].startswith("Imported 1 task")


def test_report_refuses_another_pipeline_run(
    platform: _Platform, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = platform.run
    monkeypatch.setattr(
        platform, "run", lambda args: {**original(args), "pipeline_key": "pii_clean"}
    )

    result, payload = _invoke("report", _RUN)

    assert result.exit_code == ExitCode.USAGE
    assert "not a Harbor import" in payload["message"]


def test_imports_lists_runs_into_a_taskset(platform: _Platform) -> None:
    platform.tasksets[_TASKSET] = {"id": _TASKSET, "name": "tb2", "can_edit": True}

    result, payload = _invoke("imports", "--taskset", "tb2")

    assert result.exit_code == 0, result.output
    assert payload["total"] == 1
    assert payload["runs"][0]["id"] == _RUN
    assert platform.run_queries == [
        {"pipeline_key": ["harbor_import"], "limit": ["20"], "arg": [f"taskset_id:{_TASKSET}"]}
    ]
