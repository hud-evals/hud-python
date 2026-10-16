"""``hud harbor``: import Harbor folders or bundles into a taskset, and follow imports."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.support import Reply, write_files

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import FakeServices, Hud, Request

TASKSET = "44444444-4444-4444-8444-444444444444"
FILE = "55555555-5555-4555-8555-555555555555"
IMPORT = "66666666-6666-4666-8666-666666666666"
LIVE = {"queued", "starting", "running", "cancelling"}
REPORT = {
    "schema_version": "harbor_import_report.v5",
    "taskset_id": TASKSET,
    "tasks": [
        {
            "name": "hello",
            "source": "bundle",
            "path": "tasks/hello",
            "build_version": 1,
            "status": "imported",
        }
    ],
    "jobs": [
        {
            "path": "run-1",
            "name": "run-1",
            "job_id": "job-hud-1",
            "status": "imported",
            "trials": [
                {"path": "run-1/hello__a", "status": "imported"},
                {"path": "run-1/gone__b", "status": "skipped", "reason": "unknown task"},
            ],
        }
    ],
    "warnings": [],
    "error": None,
}


def imported(status: str, *, format: str = "harbor", error: str | None = None) -> dict[str, Any]:
    """An import record at ``status`` as the platform returns it."""
    reported = status in {"completed", "stopped"}
    return {
        "id": IMPORT,
        "format": format,
        "taskset_id": TASKSET,
        "file": {"id": FILE, "filename": "tb2.zip", "size_bytes": 10},
        "status": status,
        "summary": "Imported 1 task(s) and 1 run(s)" if reported else None,
        "metrics": {"tasks_imported": 1.0} if reported else None,
        "error": error,
        "created_at": "2026-10-07T12:00:00Z",
        "completed_at": None if status in LIVE else "2026-10-07T12:01:00Z",
    }


class Importer:
    """The platform's tasksets, imports API and signed upload storage."""

    def __init__(self, services: FakeServices) -> None:
        self.services = services
        self.tasksets: dict[str, dict[str, Any]] = {}
        services.route("api", "GET", "/v2/tasksets/by-name/{name}", handler=self.by_name)
        services.route("api", "GET", "/v2/tasksets/{id}", handler=self.get)
        services.route("api", "POST", "/v2/tasksets", handler=self.create)
        upload = {
            "file": {"id": FILE, "filename": "tb2.zip"},
            "upload_url": f"{services.url('api')}/storage/upload?signature=secret",
        }
        services.route("api", "POST", "/v2/imports/uploads", json=upload)
        services.route("api", "PUT", "/storage/upload", json={})
        services.route("api", "POST", f"/v2/data/{FILE}/complete", json={"id": FILE})
        services.route("api", "DELETE", f"/v2/data/{FILE}", status=204)
        services.route("api", "GET", f"/v2/imports/{IMPORT}/report", json=REPORT)
        services.route("api", "POST", f"/v2/imports/{IMPORT}/cancel", json=imported("cancelling"))
        self.progress("completed")
        self.offer(max_input_bytes=1024**3)

    def offer(self, **harbor: Any) -> None:
        """The import formats the platform lists; none when ``harbor`` is ``None``."""
        formats = (
            []
            if harbor.get("max_input_bytes") is None
            else [{"format": "harbor", "title": "Harbor", **harbor}]
        )
        self.services.route("api", "GET", "/v2/imports/formats", json={"formats": formats})

    def progress(self, *statuses: str, **record: Any) -> None:
        """The import starts at the first status, and each later read moves on one."""
        started, *later = [imported(status, **record) for status in statuses]
        self.services.route("api", "POST", "/v2/imports", json=started)
        self.services.route(
            "api", "GET", f"/v2/imports/{IMPORT}", *[Reply(json=r) for r in later or [started]]
        )
        self.services.route("api", "GET", "/v2/imports", json={"imports": [started], "total": 1})

    def by_name(self, request: Request) -> Reply:
        for taskset in self.tasksets.values():
            if taskset["name"] == request.params["name"]:
                return Reply(json={"taskset_id": taskset["id"], "name": taskset["name"]})
        return Reply(status=404, json={"detail": "Taskset not found"})

    def get(self, request: Request) -> Reply:
        taskset = self.tasksets.get(request.params["id"])
        return Reply(json=taskset) if taskset else Reply(status=404, json={"detail": "Not found"})

    def create(self, request: Request) -> Reply:
        self.tasksets[TASKSET] = {"id": TASKSET, "name": request.json["name"], "can_edit": True}
        return Reply(json=self.tasksets[TASKSET])

    def posted(self) -> list[str]:
        return [request.path for request in self.services.requests("api", "POST")]

    def bundle(self) -> zipfile.ZipFile:
        (put,) = self.services.requests("api", "PUT", "/storage/upload")
        return zipfile.ZipFile(io.BytesIO(put.body))


@pytest.fixture
def importer(platform: FakeServices) -> Importer:
    return Importer(platform)


@pytest.fixture
def folders(tmp_path: Path) -> list[str]:
    """A tasks folder and a job folder with one trial, as Harbor writes them."""
    files = {
        "tasks/hello/task.toml": "version = '1.0'\n",
        "tasks/hello/tests/test.sh": "exit 0\n",
        "run-1/config.json": "{}",
        "run-1/result.json": "{}",
        "run-1/hello__a/config.json": "{}",
        "run-1/hello__a/result.json": "{}",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return [str(tmp_path / "tasks"), str(tmp_path / "run-1")]


def test_import_uploads_the_folders_and_imports_them_into_a_new_taskset(
    hud: Hud, importer: Importer, folders: list[str]
) -> None:
    result = hud("harbor", "import", *folders, "--taskset", "TB2 Sample", "--yes", "--json")

    assert result.exit_code == 0, result
    assert importer.tasksets[TASKSET]["name"] == "tb2-sample"
    assert sorted(importer.bundle().namelist()) == [
        "run-1/config.json",
        "run-1/hello__a/config.json",
        "run-1/hello__a/result.json",
        "run-1/result.json",
        "tasks/hello/task.toml",
        "tasks/hello/tests/test.sh",
    ]
    assert importer.services.bodies("api", "POST", "/v2/imports") == [
        {"format": "harbor", "file_id": FILE, "taskset_id": TASKSET}
    ]
    assert (result.json["taskset"]["created"], result.json["import"]["status"]) == (
        True,
        "completed",
    )
    trials = result.json["report"]["jobs"][0]["trials"]
    assert [trial["status"] for trial in trials] == ["imported", "skipped"]
    assert "signature" not in str(result)


def test_rerunning_an_import_reuses_the_taskset_its_name_created(
    hud: Hud, importer: Importer, folders: list[str]
) -> None:
    first = hud("harbor", "import", *folders, "--taskset", "TB2/Sample", "--yes", "--json")
    second = hud("harbor", "import", *folders, "--taskset", "TB2/Sample", "--yes", "--json")

    assert (first.exit_code, second.exit_code) == (0, 0), second
    assert importer.posted().count("/v2/tasksets") == 1
    assert second.json["taskset"] == {"id": TASKSET, "name": "tb2sample", "created": False}


def test_import_link_saves_the_taskset_as_the_directory_default(
    hud: Hud, importer: Importer, folders: list[str]
) -> None:
    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--link")

    assert result.exit_code == 0, result
    assert json.loads((hud.cwd / ".hud" / "config.json").read_text())["taskset_id"] == TASKSET


def test_import_uploads_a_given_zip_as_it_is(hud: Hud, importer: Importer, tmp_path: Path) -> None:
    bundle = tmp_path / "prepared.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("hello/task.toml", "version = '1.0'\n")

    result = hud("harbor", "import", str(bundle), "--yes", "--json")

    assert result.exit_code == 0, result
    (put,) = importer.services.requests("api", "PUT", "/storage/upload")
    assert put.body == bundle.read_bytes()
    assert importer.tasksets[TASKSET]["name"] == "prepared"


def test_a_dry_run_plans_without_writing(hud: Hud, importer: Importer, folders: list[str]) -> None:
    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert (result.json["taskset"]["created"], result.json["bundle"]["files"]) == (True, 6)
    assert importer.posted() == []


@pytest.mark.parametrize(
    ("setup", "argv", "exit_code", "error"),
    [
        pytest.param(
            lambda importer: importer.tasksets.update(
                {TASKSET: {"id": TASKSET, "name": "theirs", "can_edit": False}}
            ),
            ["--taskset", TASKSET],
            1,
            "permission_denied",
            id="a-taskset-the-caller-cannot-edit",
        ),
        pytest.param(
            lambda importer: importer.offer(max_input_bytes=10),
            ["--taskset", "tb2"],
            2,
            "usage",
            id="a-bundle-over-the-limit",
        ),
    ],
)
def test_import_refuses_before_uploading(
    hud: Hud,
    importer: Importer,
    folders: list[str],
    setup: Any,
    argv: list[str],
    exit_code: int,
    error: str,
) -> None:
    setup(importer)

    result = hud("harbor", "import", *folders, *argv, "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (exit_code, error), result
    assert importer.posted() == []


def test_a_refused_upload_deletes_its_reservation_and_imports_nothing(
    hud: Hud, importer: Importer, folders: list[str]
) -> None:
    importer.services.route("api", "PUT", "/storage/upload", status=403)

    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--json")

    assert result.exit_code == 1, result
    assert [r.path for r in importer.services.requests("api", "DELETE")] == [f"/v2/data/{FILE}"]
    assert "/v2/imports" not in importer.posted()
    assert "signature" not in str(result)


@pytest.mark.parametrize(
    ("statuses", "report", "exit_code", "reported"),
    [
        pytest.param(["queued", "running", "completed"], REPORT, 0, True, id="completed"),
        pytest.param(
            ["stopped"],
            {**REPORT, "error": {"step": "add", "message": "boom"}},
            1,
            True,
            id="stopped",
        ),
        pytest.param(["failed"], REPORT, 1, False, id="failed"),
    ],
)
def test_import_waits_for_the_import_and_fails_unless_it_completed(
    hud: Hud,
    importer: Importer,
    folders: list[str],
    statuses: list[str],
    report: dict[str, Any] | None,
    exit_code: int,
    reported: bool,
) -> None:
    importer.progress(*statuses, error="crashed" if statuses == ["failed"] else None)
    importer.services.route(
        "api", "GET", f"/v2/imports/{IMPORT}/report", json=report, status=200 if report else 404
    )

    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--json")

    assert result.exit_code == exit_code, result
    assert result.json["import"]["status"] == statuses[-1]
    shown = result.json["report"]
    assert (shown is not None) is reported
    assert (shown or {}).get("error") == (report or {}).get("error")


def test_no_wait_returns_the_started_import(
    hud: Hud, importer: Importer, folders: list[str]
) -> None:
    importer.progress("queued")

    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--no-wait", "--json")

    assert (result.exit_code, result.json["import"]["status"]) == (0, "queued"), result
    assert importer.services.requests("api", "GET", f"/v2/imports/{IMPORT}") == []


@pytest.mark.parametrize(
    ("argv", "statuses", "report", "version"),
    [
        pytest.param(
            ["--wait"],
            ["queued", "running", "completed"],
            REPORT,
            "harbor_import_report.v5",
            id="waited",
        ),
        pytest.param(
            [],
            ["completed"],
            {"schema_version": "harbor_import_report.v9"},
            None,
            id="unknown-schema",
        ),
    ],
)
def test_report_shows_an_ended_imports_report(
    hud: Hud,
    importer: Importer,
    argv: list[str],
    statuses: list[str],
    report: dict[str, Any],
    version: str | None,
) -> None:
    importer.progress(statuses[0], *statuses)
    importer.services.route("api", "GET", f"/v2/imports/{IMPORT}/report", json=report)

    result = hud("harbor", "report", IMPORT, *argv, "--json")

    assert result.exit_code == 0, result
    shown_version = (result.json["report"] or {}).get("schema_version")
    assert (result.json["import"]["status"], shown_version) == (statuses[-1], version)


def test_report_refuses_an_import_of_another_format(hud: Hud, platform: FakeServices) -> None:
    Importer(platform).progress("completed", format="parquet")

    result = hud("harbor", "report", IMPORT, "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result


@pytest.mark.parametrize(("status", "cancelled"), [("running", True), ("completed", False)])
def test_cancel_stops_only_a_live_import(
    hud: Hud, importer: Importer, status: str, cancelled: bool
) -> None:
    importer.progress(status)

    result = hud("harbor", "cancel", IMPORT, "--yes", "--json")

    assert result.exit_code == 0, result
    assert (f"/v2/imports/{IMPORT}/cancel" in importer.posted()) is cancelled


def test_imports_lists_a_tasksets_harbor_imports(hud: Hud, importer: Importer) -> None:
    importer.tasksets[TASKSET] = {"id": TASKSET, "name": "tb2", "can_edit": True}

    result = hud("harbor", "imports", "--taskset", "tb2", "--json")

    assert result.exit_code == 0, result
    assert [item["id"] for item in result.json["imports"]] == [IMPORT]
    (listed,) = importer.services.requests("api", "GET", "/v2/imports")
    assert listed.query == {"limit": ["20"], "format": ["harbor"], "taskset_id": [TASKSET]}


def test_imports_of_an_unknown_taskset_lists_nothing(hud: Hud, importer: Importer) -> None:
    result = hud("harbor", "imports", "--taskset", TASKSET, "--json")

    assert result.exit_code == 1, result
    assert importer.services.requests("api", "GET", "/v2/imports") == []


def harbor_task(root: Path) -> Path:
    """A minimal Harbor task folder, ``hello``, under ``root``."""
    files = {"task.toml": "version = '1.0'\n", "environment/Dockerfile": "FROM ubuntu\n"}
    write_files(root / "hello", files)
    return root / "hello"


def test_import_bundles_linked_files_by_content_and_skips_linked_folders(
    hud: Hud, importer: Importer, tmp_path: Path
) -> None:
    task = harbor_task(tmp_path / "src")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "data.txt").write_text("shared")
    (task / "environment" / "data.txt").symlink_to(shared / "data.txt")
    (task / "environment" / "linked").symlink_to(shared)
    (task / "dangling").symlink_to(tmp_path / "missing")

    result = hud("harbor", "import", str(task), "--taskset", "tb2", "--yes")

    assert result.exit_code == 0, result
    bundle = importer.bundle()
    assert bundle.read("hello/environment/data.txt") == b"shared"
    assert not any("linked" in name or "dangling" in name for name in bundle.namelist())


def test_import_refuses_two_folders_with_one_name_before_any_request(
    hud: Hud, importer: Importer, tmp_path: Path
) -> None:
    first, second = harbor_task(tmp_path / "a"), harbor_task(tmp_path / "b")

    result = hud("harbor", "import", str(first), str(second), "--yes")

    assert result.exit_code != 0, result
    assert importer.posted() == []
