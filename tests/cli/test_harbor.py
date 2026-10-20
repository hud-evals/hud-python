"""``hud harbor import``: bundle Harbor folders into a taskset and follow the import to its
end."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import write_files
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from tests.conftest import Hud
    from tests.fixtures.services import FakeServices

IMPORT_ID = "66666666-6666-4666-8666-666666666666"
FILE_ID = "ffffffff-0000-4000-8000-000000000001"
TASKSET_ID = "eeeeeeee-0000-4000-8000-000000000001"
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
