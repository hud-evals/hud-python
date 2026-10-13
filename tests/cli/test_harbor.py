"""``hud harbor``: import Harbor folders or bundles into a taskset, and follow imports.

The CLI runs as a subprocess against the fake platform, which hosts a stateful
importer: it answers the imports API, keeps tasksets, and stores the uploaded
bundle the way signed storage would."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlsplit

import pytest

from hud.cli import ExitCode
from tests.harness import Reply

from .conftest import WEB_URL

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import FakeServices, Hud, Request, Result

_USER = "11111111-1111-4111-8111-111111111111"
_TEAM = "22222222-2222-4222-8222-222222222222"
_TASKSET = "44444444-4444-4444-8444-444444444444"
_FILE = "55555555-5555-4555-8555-555555555555"
_IMPORT = "66666666-6666-4666-8666-666666666666"
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


class NotFound(Exception):
    """The importer has no such resource; the platform answers 404."""


class _Platform:
    """The platform and storage endpoints a Harbor import touches."""

    def __init__(self, storage_url: str) -> None:
        self.storage_url = storage_url
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
        self.upload_headers: dict[str, str] | None = None

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

    def request(
        self, method: str, url: str, request_query: dict[str, list[str]], **kwargs: Any
    ) -> Any:
        parts = urlsplit(url)
        path = parts.path.removeprefix("/v2")
        self.calls.append((method, path, kwargs.get("json")))
        if path == "/auth/me":
            return {"user_id": _USER, "team_id": _TEAM}
        if path.startswith("/imports") and not self.has_imports_api:
            raise NotFound("Not Found")
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
            raise NotFound("Taskset not found")
        if method == "GET" and path.startswith("/tasksets/"):
            taskset_id = path.removeprefix("/tasksets/")
            if taskset_id not in self.tasksets:
                raise NotFound("Taskset not found")
            return self.tasksets[taskset_id]
        if method == "POST" and path == "/tasksets":
            created = {"id": _TASKSET, "name": kwargs["json"]["name"], "can_edit": True}
            self.tasksets[_TASKSET] = created
            return created
        if method == "POST" and path == "/imports/uploads":
            filename = kwargs["json"]["filename"]
            return {"file": {"id": _FILE, "filename": filename}, "upload_url": self.storage_url}
        if method == "POST" and path == f"/data/{_FILE}/complete":
            return {"id": _FILE, "filename": "tb2.zip", "size_bytes": 10}
        if method == "DELETE" and path == f"/data/{_FILE}":
            return None
        if method == "POST" and path == "/imports":
            return self.current_import()
        if path == "/imports":
            self.list_queries.append(dict(request_query))
            return {"imports": [self.current_import()], "total": 1}
        if path == f"/imports/{_IMPORT}":
            return self.current_import()
        if path == f"/imports/{_IMPORT}/report":
            if self.report is None:
                raise NotFound("Import report not found")
            return self.report
        if path == f"/imports/{_IMPORT}/cancel":
            return self.imported("cancelling")
        if path.startswith("/imports/"):
            raise NotFound("Import not found")
        raise AssertionError((method, url))

    def handle(self, request: Request) -> Reply:
        """Answer one platform API request."""
        url = request.path + (f"?{urlencode(request.query, doseq=True)}" if request.query else "")
        try:
            body = self.request(request.method, url, request.query, json=request.json)
        except NotFound as error:
            return Reply(status=404, json={"detail": str(error)})
        return Reply(status=204) if body is None else Reply(json=body)

    def store(self, request: Request) -> Reply:
        """Signed storage: keep the uploaded bundle and its headers."""
        self.uploaded = request.body
        self.upload_headers = request.headers
        return Reply(status=self.storage_status)


@pytest.fixture
def importer(platform: FakeServices) -> _Platform:
    """The platform's imports API and signed storage, answered by one stateful fake."""
    fake = _Platform(f"{platform.url('api')}/storage/upload?signature=secret")
    for method in ("GET", "POST", "DELETE"):
        platform.route("api", method, "/v2/{rest:path}", handler=fake.handle)
    platform.route("api", "PUT", "/storage/upload", handler=fake.store)
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


def run(hud: Hud, *args: str) -> tuple[Result, dict[str, Any]]:
    result = hud("harbor", *args, "--json")
    return result, json.loads(result.stdout)


def test_import_creates_a_new_taskset_and_imports_into_it(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    result, payload = run(hud, "import", *folders, "--taskset", "TB2 Sample", "--yes")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert importer.body("POST", "/tasksets") == {"name": "tb2-sample", "project_id": None}
    assert importer.body("POST", "/imports") == {
        "format": "harbor",
        "file_id": _FILE,
        "taskset_id": _TASKSET,
    }
    assert importer.paths("POST")[-2:] == [f"/data/{_FILE}/complete", "/imports"]
    assert payload["taskset"] == {"id": _TASKSET, "name": "tb2-sample", "created": True}
    assert payload["import"]["status"] == "completed"
    assert payload["import"]["summary"] == "Imported 1 task(s) and 1 run(s)"
    assert payload["report"]["taskset_id"] == _TASKSET
    assert [trial["status"] for trial in payload["report"]["jobs"][0]["trials"]] == [
        "imported",
        "skipped",
    ]


def test_import_uploads_each_folder_under_its_name(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    result, _ = run(hud, "import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == 0, result.stdout + result.stderr
    reservation = importer.body("POST", "/imports/uploads")
    assert reservation["format"] == "harbor"
    assert reservation["filename"] == "tb2.zip"
    assert reservation["content_type"] == "application/zip"
    assert importer.upload_headers is not None
    assert importer.upload_headers["content-length"] == str(reservation["size_bytes"])
    assert importer.uploaded is not None
    with zipfile.ZipFile(io.BytesIO(importer.uploaded)) as bundle:
        assert sorted(bundle.namelist()) == [
            "run-1/config.json",
            "run-1/hello__a/config.json",
            "run-1/hello__a/result.json",
            "run-1/result.json",
            "tasks/hello/task.toml",
            "tasks/hello/tests/test.sh",
        ]


def test_import_defaults_to_the_taskset_named_after_the_first_folder(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.tasksets[_TASKSET] = {"id": _TASKSET, "name": "tasks", "can_edit": True}

    result, payload = run(hud, "import", *folders, "--yes")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "/tasksets" not in importer.paths("POST")
    assert payload["taskset"] == {"id": _TASKSET, "name": "tasks", "created": False}


def test_rerunning_an_import_finds_the_taskset_its_name_created(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    first, _ = run(hud, "import", *folders, "--taskset", "TB2/Sample", "--yes")
    second, payload = run(hud, "import", *folders, "--taskset", "TB2/Sample", "--yes")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert importer.paths("POST").count("/tasksets") == 1
    assert importer.paths("GET").count("/tasksets/by-name/tb2sample") == 2
    assert payload["taskset"] == {"id": _TASKSET, "name": "tb2sample", "created": False}


def test_import_refuses_an_unknown_taskset_id_instead_of_creating_one(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    result, payload = run(hud, "import", *folders, "--taskset", _TASKSET, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert payload["message"] == f"Taskset {_TASKSET} is inaccessible or deleted."
    assert importer.paths("POST") == []


def test_import_link_saves_the_taskset_as_the_directory_default(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    result, _ = run(hud, "import", *folders, "--taskset", "tb2", "--yes", "--link")

    assert result.exit_code == 0, result.stdout + result.stderr
    link = json.loads((hud.cwd / ".hud" / "config.json").read_text())
    assert link["taskset_id"] == _TASKSET


def test_import_refuses_a_taskset_the_caller_cannot_edit_before_uploading(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.tasksets[_TASKSET] = {"id": _TASKSET, "name": "theirs", "can_edit": False}

    result, payload = run(hud, "import", *folders, "--taskset", _TASKSET, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "permission_denied"
    assert "/imports/uploads" not in importer.paths("POST")


def test_import_refuses_a_bundle_over_the_format_limit(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.max_input_bytes = 10

    result, payload = run(hud, "import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.USAGE
    assert payload["message"].endswith("a Harbor import takes at most 10 bytes.")
    assert importer.paths("POST") == []


def test_dry_run_plans_without_writing(hud: Hud, importer: _Platform, folders: list[str]) -> None:
    result, payload = run(hud, "import", *folders, "--taskset", "tb2", "--dry-run")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload["dry_run"] is True
    assert payload["taskset"] == {"id": None, "name": "tb2", "created": True}
    assert payload["bundle"]["filename"] == "tb2.zip"
    assert payload["bundle"]["files"] == 6
    assert importer.paths("POST") == []


def test_a_failed_upload_deletes_its_reservation_without_leaking_the_signed_url(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.storage_status = 403

    result, payload = run(hud, "import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["message"] == "Storage refused the upload: HTTP 403"
    assert importer.paths("DELETE") == [f"/data/{_FILE}"]
    assert "/imports" not in importer.paths("POST")
    assert "signature" not in (result.stdout + result.stderr)


def test_a_stopped_import_exits_with_failure_and_its_report(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.import_statuses = ["stopped"]
    importer.report = _report(error={"step": "add the tasks to the taskset", "message": "boom"})

    result, payload = run(hud, "import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["import"]["status"] == "stopped"
    assert payload["report"]["error"] == {"step": "add the tasks to the taskset", "message": "boom"}


def test_a_failed_import_without_a_report_exits_with_its_error(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.import_statuses = ["failed"]
    importer.import_error = "the importer crashed"

    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert "The import failed: the importer crashed" in (result.stdout + result.stderr)
    assert f"/imports/{_IMPORT}/report" not in importer.paths("GET")


def test_a_stopped_import_without_a_report_shows_its_summary(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.import_statuses = ["stopped"]
    importer.report = None

    result, payload = run(hud, "import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["report"] is None
    assert payload["import"]["summary"] == "Imported 1 task(s) and 1 run(s)"


def test_no_wait_returns_the_started_import(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.import_statuses = ["queued"]

    result, payload = run(hud, "import", *folders, "--taskset", "tb2", "--yes", "--no-wait")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload["import"]["status"] == "queued"
    assert payload["report"] is None
    assert f"/imports/{_IMPORT}" not in importer.paths("GET")


def test_import_uploads_a_given_zip_as_it_is(hud: Hud, importer: _Platform, tmp_path: Path) -> None:
    bundle = tmp_path / "prepared.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("hello/task.toml", "version = '1.0'\n")

    result, payload = run(hud, "import", str(bundle), "--yes")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert importer.uploaded == bundle.read_bytes()
    assert payload["taskset"]["name"] == "prepared"


def test_import_packs_a_folder_named_like_a_zip(
    hud: Hud, importer: _Platform, tmp_path: Path
) -> None:
    folder = tmp_path / "release.zip"
    folder.mkdir()
    (folder / "task.toml").write_text("version = '1.0'\n")

    result, payload = run(hud, "import", str(folder), "--taskset", "tb2", "--yes")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload["bundle"]["files"] == 1
    assert importer.uploaded is not None
    with zipfile.ZipFile(io.BytesIO(importer.uploaded)) as bundle:
        assert bundle.namelist() == ["release.zip/task.toml"]


def test_import_refuses_a_zip_mixed_with_folders_before_any_request(
    hud: Hud, importer: _Platform, folders: list[str], tmp_path: Path
) -> None:
    bundle = tmp_path / "prepared.zip"
    bundle.write_bytes(b"zip")

    result, payload = run(hud, "import", str(bundle), folders[0], "--yes")

    assert result.exit_code == ExitCode.USAGE
    assert payload["message"] == "Pass either one .zip bundle or folders, not both or several zips."
    assert importer.calls == []


def test_import_fails_where_the_platform_offers_no_harbor_import(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.offers_harbor_import = False

    result, payload = run(hud, "import", *folders, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert payload["message"].startswith("Harbor import is not available on ")


def test_import_fails_where_the_platform_has_no_imports(
    hud: Hud, importer: _Platform, folders: list[str]
) -> None:
    importer.has_imports_api = False

    result, payload = run(hud, "import", *folders, "--yes")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert payload["message"].startswith("Harbor import is not available on ")


def test_import_prints_the_report_tables(hud: Hud, importer: _Platform, folders: list[str]) -> None:
    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "1 imported, 1 skipped" in (result.stdout + result.stderr)
    assert "it ran a task that is neither in the bundle nor in HUD" in (
        result.stdout + result.stderr
    )
    assert f"{WEB_URL}/tasksets/{_TASKSET}/imports" in (result.stdout + result.stderr)


def test_report_wait_follows_the_import_until_it_ends(hud: Hud, importer: _Platform) -> None:
    importer.import_statuses = ["queued", "running", "completed"]

    result, payload = run(hud, "report", _IMPORT, "--wait")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload["import"]["status"] == "completed"
    assert payload["report"]["taskset_id"] == _TASKSET


def test_report_of_a_running_import_without_wait_has_no_report(
    hud: Hud, importer: _Platform
) -> None:
    importer.import_statuses = ["running"]

    result, payload = run(hud, "report", _IMPORT)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload["report"] is None
    assert f"/imports/{_IMPORT}/report" not in importer.paths("GET")


def test_report_shows_tasks_an_import_updated(hud: Hud, importer: _Platform) -> None:
    [task] = importer.report["tasks"]
    importer.report["tasks"] = [{**task, "build_version": 2, "status": "updated"}]

    result = hud("harbor", "report", _IMPORT)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "updated" in (result.stdout + result.stderr)


def test_report_reads_a_v4_report(hud: Hud, importer: _Platform) -> None:
    importer.report = _report(schema_version="harbor_import_report.v4")

    result, payload = run(hud, "report", _IMPORT)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload["report"]["schema_version"] == "harbor_import_report.v4"
    assert payload["report"]["tasks"][0]["status"] == "imported"


def test_report_of_an_unknown_schema_falls_back_to_the_summary(
    hud: Hud, importer: _Platform
) -> None:
    importer.report = {"schema_version": "harbor_import_report.v9"}

    result, payload = run(hud, "report", _IMPORT)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert payload["report"] is None
    assert payload["import"]["summary"] == "Imported 1 task(s) and 1 run(s)"


def test_report_refuses_an_import_of_another_format(hud: Hud, importer: _Platform) -> None:
    importer.import_format = "parquet"

    result, payload = run(hud, "report", _IMPORT)

    assert result.exit_code == ExitCode.USAGE
    assert payload["message"] == f"Import {_IMPORT} is a parquet import, not a Harbor import."


def test_report_of_an_unknown_import_is_not_found(hud: Hud, importer: _Platform) -> None:
    result, payload = run(hud, "report", "88888888-8888-4888-8888-888888888888")

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"


def test_cancel_requests_cancellation_of_a_running_import(hud: Hud, importer: _Platform) -> None:
    importer.import_statuses = ["running"]

    result, payload = run(hud, "cancel", _IMPORT, "--yes")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert f"/imports/{_IMPORT}/cancel" in importer.paths("POST")
    assert payload["status"] == "cancelling"


def test_cancel_leaves_an_ended_import_alone(hud: Hud, importer: _Platform) -> None:
    result, payload = run(hud, "cancel", _IMPORT, "--yes")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert importer.paths("POST") == []
    assert payload["status"] == "completed"


def test_imports_filters_by_taskset(hud: Hud, importer: _Platform) -> None:
    importer.tasksets[_TASKSET] = {"id": _TASKSET, "name": "tb2", "can_edit": True}

    result, payload = run(hud, "imports", "--taskset", "tb2")

    assert result.exit_code == 0, result.stdout + result.stderr
    assert [item["id"] for item in payload["imports"]] == [_IMPORT]
    assert importer.list_queries == [
        {"limit": ["20"], "format": ["harbor"], "taskset_id": [_TASKSET]}
    ]


def test_imports_refuses_an_unknown_taskset_id(hud: Hud, importer: _Platform) -> None:
    result, payload = run(hud, "imports", "--taskset", _TASKSET)

    assert result.exit_code == ExitCode.FAILURE
    assert payload["error"] == "not_found"
    assert importer.list_queries == []


def test_imports_refuses_a_limit_the_platform_does_not_page(hud: Hud, importer: _Platform) -> None:
    result = hud("harbor", "imports", "--limit", "101")

    assert result.exit_code == ExitCode.USAGE
    assert importer.calls == []
