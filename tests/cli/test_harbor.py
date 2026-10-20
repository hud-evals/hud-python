"""``hud harbor``: import Harbor folders or a bundle into a taskset, follow the import, and
read what it did."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import write_files
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices, Request

IMPORT_ID = "66666666-6666-4666-8666-666666666666"
FILE_ID = "ffffffff-0000-4000-8000-000000000001"
TASKSET_ID = "eeeeeeee-0000-4000-8000-000000000001"
OTHER_TASKSET_ID = "eeeeeeee-0000-4000-8000-000000000002"
REPORT = {
    "schema_version": "harbor_import_report.v5",
    "taskset_id": TASKSET_ID,
    "tasks": [
        {
            "name": "hello",
            "source": "bundle",
            "path": "tasks/hello",
            "build_version": 1,
            "status": "imported",
        }
    ],
    "jobs": [],
    "warnings": [],
    "error": None,
}
FOLDERS = {
    "tasks/hello/task.toml": "version = '1.0'\n",
    "tasks/hello/tests/test.sh": "exit 0\n",
    "run-1/config.json": "{}",
    "run-1/result.json": "{}",
}


def record(status: str, *, error: str | None = None, format: str = "harbor") -> dict[str, Any]:
    """An import as the platform reports it at ``status``."""
    ended = status not in {"queued", "starting", "running", "cancelling"}
    return {
        "id": IMPORT_ID,
        "format": format,
        "taskset_id": TASKSET_ID,
        "file": {"id": FILE_ID, "filename": "bundle.zip", "size_bytes": 1},
        "status": status,
        "summary": "Imported 1 task(s)" if status in {"completed", "stopped"} else None,
        "metrics": None,
        "error": error,
        "created_at": "2026-10-07T12:00:00Z",
        "completed_at": "2026-10-07T12:01:00Z" if ended else None,
    }


@pytest.fixture
def importer(signed_in: FakeServices, hud: Hud) -> FakeServices:
    """Harbor folders in the directory, and a platform that stores a bundle, creates the
    ``tb2`` taskset it is not holding yet, and completes each import with ``REPORT``."""
    write_files(hud.cwd, FOLDERS)
    formats = [{"format": "harbor", "title": "Harbor", "max_input_bytes": 1024**3}]
    signed_in.route("api", "GET", "/v2/imports/formats", json={"formats": formats})
    signed_in.route(
        "api",
        "POST",
        "/v2/imports/uploads",
        json={
            "file": {"id": FILE_ID, "filename": "bundle.zip"},
            "upload_url": f"{signed_in.url('api')}/uploads/{FILE_ID}?signature=secret",
        },
    )
    signed_in.route("api", "PUT", "/uploads/{id}", json={})
    signed_in.route("api", "POST", "/v2/data/{id}/complete", json={"id": FILE_ID})
    signed_in.route("api", "DELETE", "/v2/data/{id}", status=204)
    signed_in.route("api", "GET", "/v2/tasksets/by-name/{name}", status=404, json={"detail": "No"})
    signed_in.route(
        "api", "POST", "/v2/tasksets", json={"id": TASKSET_ID, "name": "tb2", "can_edit": True}
    )
    signed_in.route("api", "POST", "/v2/imports", json=record("completed"))
    signed_in.route("api", "GET", f"/v2/imports/{IMPORT_ID}/report", json=REPORT)
    return signed_in


def bundle(services: FakeServices) -> dict[str, bytes]:
    """The bundle the platform received, member by member."""
    (upload,) = services.requests("api", "PUT", "/uploads/{id}")
    with zipfile.ZipFile(io.BytesIO(upload.body)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_folders_are_zipped_by_their_own_paths_and_imported_into_the_new_taskset(
    importer: FakeServices, hud: Hud
) -> None:
    result = hud("harbor", "import", "tasks", "run-1", "--taskset", "TB2", "--yes", "--json")

    assert result.exit_code == 0, result
    assert bundle(importer) == {name: text.encode() for name, text in FOLDERS.items()}
    assert importer.bodies("api", "POST", "/v2/tasksets")[0]["name"] == "tb2"
    assert importer.bodies("api", "POST", "/v2/imports")[0]["taskset_id"] == TASKSET_ID


def test_a_prepared_zip_is_sent_as_it_is_into_a_taskset_named_after_it(
    importer: FakeServices, hud: Hud
) -> None:
    with zipfile.ZipFile(hud.cwd / "prepared.zip", "w") as archive:
        archive.writestr("hello/task.toml", "version = '1.0'\n")

    result = hud("harbor", "import", "prepared.zip", "--yes", "--json")

    assert result.exit_code == 0, result
    assert bundle(importer) == {"hello/task.toml": b"version = '1.0'\n"}
    assert importer.bodies("api", "POST", "/v2/tasksets")[0]["name"] == "prepared"


def test_a_linked_file_is_sent_by_content_and_a_linked_folder_is_skipped(
    importer: FakeServices, hud: Hud, tmp_path: Path
) -> None:
    task = tmp_path / "src" / "hello"
    write_files(task, {"task.toml": "version = '1.0'\n", "environment/Dockerfile": "FROM ubuntu\n"})
    write_files(tmp_path / "shared", {"data.txt": "shared"})
    (task / "environment" / "data.txt").symlink_to(tmp_path / "shared" / "data.txt")
    (task / "environment" / "linked").symlink_to(tmp_path / "shared")
    (task / "dangling").symlink_to(tmp_path / "missing")

    result = hud("harbor", "import", str(task), "--taskset", "tb2", "--yes", "--json")

    assert result.exit_code == 0, result
    assert bundle(importer) == {
        "hello/task.toml": b"version = '1.0'\n",
        "hello/environment/Dockerfile": b"FROM ubuntu\n",
        "hello/environment/data.txt": b"shared",
    }


@pytest.mark.parametrize(
    ("flags", "linked"), [([], False), (["--link"], True)], ids=["kept", "linked"]
)
def test_an_import_into_an_existing_taskset_creates_none_and_links_only_when_asked(
    flags: list[str], linked: bool, importer: FakeServices, hud: Hud
) -> None:
    importer.route(
        "api", "GET", "/v2/tasksets/by-name/tb2", json={"taskset_id": TASKSET_ID, "name": "tb2"}
    )
    importer.route(
        "api",
        "GET",
        f"/v2/tasksets/{TASKSET_ID}",
        json={"id": TASKSET_ID, "name": "tb2", "can_edit": True},
    )

    result = hud("harbor", "import", "tasks", "--taskset", "tb2", *flags, "--yes", "--json")

    assert (result.exit_code, result.json["taskset"]["created"]) == (0, False), result
    assert importer.requests("api", "POST", "/v2/tasksets") == []
    link = hud.cwd / ".hud" / "config.json"
    assert (link.exists() and json.loads(link.read_text())["taskset_id"] == TASKSET_ID) is linked


def test_a_dry_run_plans_the_import_and_uploads_nothing(importer: FakeServices, hud: Hud) -> None:
    result = hud("harbor", "import", "tasks", "--taskset", "tb2", "--dry-run", "--json")

    assert (result.exit_code, result.json["dry_run"], result.json["taskset"]["created"]) == (
        0,
        True,
        True,
    ), result
    assert importer.requests("api", "POST") == importer.requests("api", "PUT") == []


def test_a_taskset_the_caller_cannot_edit_is_refused_before_uploading(
    importer: FakeServices, hud: Hud
) -> None:
    importer.route(
        "api",
        "GET",
        "/v2/tasksets/by-name/theirs",
        json={"taskset_id": TASKSET_ID, "name": "theirs"},
    )
    importer.route(
        "api",
        "GET",
        f"/v2/tasksets/{TASKSET_ID}",
        json={"id": TASKSET_ID, "name": "theirs", "can_edit": False},
    )

    result = hud("harbor", "import", "tasks", "--taskset", "theirs", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "permission_denied"), result
    assert importer.requests("api", "PUT") == []


def test_a_bundle_over_the_platforms_limit_is_refused_before_uploading(
    importer: FakeServices, hud: Hud
) -> None:
    formats = [{"format": "harbor", "title": "Harbor", "max_input_bytes": 10}]
    importer.route("api", "GET", "/v2/imports/formats", json={"formats": formats})

    result = hud("harbor", "import", "tasks", "--taskset", "tb2", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert importer.requests("api", "PUT") == []


def test_a_refused_upload_deletes_its_reservation_and_starts_no_import(
    importer: FakeServices, hud: Hud
) -> None:
    importer.route("api", "PUT", "/uploads/{id}", status=403)

    result = hud("harbor", "import", "tasks", "--taskset", "tb2", "--yes", "--json")

    assert result.exit_code == 1, result
    assert [r.path for r in importer.requests("api", "DELETE")] == [f"/v2/data/{FILE_ID}"]
    assert importer.requests("api", "POST", "/v2/imports") == []
    assert "signature" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("statuses", "error", "exit_code"),
    [
        (["queued", "running", "completed"], None, 0),
        (["stopped"], None, 1),
        (["failed"], "crashed", 1),
    ],
    ids=["completed", "stopped", "failed"],
)
def test_an_import_is_followed_to_its_end_and_fails_unless_it_completed(
    statuses: list[str], error: str | None, exit_code: int, importer: FakeServices, hud: Hud
) -> None:
    first, *later = [record(status, error=error) for status in statuses]
    importer.route("api", "POST", "/v2/imports", json=first)
    importer.route(
        "api", "GET", f"/v2/imports/{IMPORT_ID}", *[Reply(json=r) for r in later or [first]]
    )

    result = hud("harbor", "import", "tasks", "--taskset", "tb2", "--yes", "--json")

    assert (result.exit_code, result.json["import"]["status"]) == (exit_code, statuses[-1]), result


def test_no_wait_returns_the_import_as_it_started(importer: FakeServices, hud: Hud) -> None:
    importer.route("api", "POST", "/v2/imports", json=record("queued"))

    result = hud("harbor", "import", "tasks", "--taskset", "tb2", "--no-wait", "--yes", "--json")

    assert (result.exit_code, result.json["import"]["status"], result.json["report"]) == (
        0,
        "queued",
        None,
    ), result


def test_report_of_an_unknown_schema_is_left_out(signed_in: FakeServices, hud: Hud) -> None:
    signed_in.route("api", "GET", f"/v2/imports/{IMPORT_ID}", json=record("completed"))
    signed_in.route(
        "api",
        "GET",
        f"/v2/imports/{IMPORT_ID}/report",
        json={"schema_version": "harbor_import_report.v9"},
    )

    result = hud("harbor", "report", IMPORT_ID, "--json")

    assert (result.exit_code, result.json["report"]) == (0, None), result


def test_report_refuses_an_import_of_another_format(signed_in: FakeServices, hud: Hud) -> None:
    signed_in.route(
        "api", "GET", f"/v2/imports/{IMPORT_ID}", json=record("completed", format="parquet")
    )

    result = hud("harbor", "report", IMPORT_ID, "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result


@pytest.mark.parametrize(
    ("status", "cancelled"), [("running", True), ("completed", False)], ids=["live", "ended"]
)
def test_cancel_stops_a_live_import_and_leaves_an_ended_one(
    status: str, cancelled: bool, signed_in: FakeServices, hud: Hud
) -> None:
    signed_in.route("api", "GET", f"/v2/imports/{IMPORT_ID}", json=record(status))
    signed_in.route("api", "POST", f"/v2/imports/{IMPORT_ID}/cancel", json=record("cancelling"))

    result = hud("harbor", "cancel", IMPORT_ID, "--yes", "--json")

    assert result.exit_code == 0, result
    assert bool(signed_in.requests("api", "POST", f"/v2/imports/{IMPORT_ID}/cancel")) is cancelled


def test_imports_lists_a_tasksets_harbor_imports(signed_in: FakeServices, hud: Hud) -> None:
    signed_in.route(
        "api", "GET", "/v2/tasksets/by-name/tb2", json={"taskset_id": TASKSET_ID, "name": "tb2"}
    )
    signed_in.route(
        "api", "GET", f"/v2/tasksets/{TASKSET_ID}", json={"id": TASKSET_ID, "name": "tb2"}
    )
    imports = [
        record("running"),
        {**record("running"), "id": f"{IMPORT_ID[:-1]}7", "taskset_id": OTHER_TASKSET_ID},
        {**record("running", format="parquet"), "id": f"{IMPORT_ID[:-1]}8"},
    ]

    def filtered(request: Request) -> Reply:
        matching = [
            item
            for item in imports
            if [item["format"]] == request.query.get("format", [item["format"]])
            and [item["taskset_id"]] == request.query.get("taskset_id", [item["taskset_id"]])
        ]
        return Reply(json={"imports": matching, "total": len(matching)})

    signed_in.route("api", "GET", "/v2/imports", handler=filtered)

    listed = hud("harbor", "imports", "--taskset", "tb2", "--json")

    assert listed.exit_code == 0, listed
    assert [item["id"] for item in listed.json["imports"]] == [IMPORT_ID]
