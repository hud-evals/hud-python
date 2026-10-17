"""``hud harbor``: import Harbor folders or bundles into a taskset, and follow imports."""

from __future__ import annotations

import io
import json
import zipfile
from typing import TYPE_CHECKING, Any

import pytest

from tests.support import write_files
from tests.support.platform import Import

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import Hud, Platform

IMPORT = "66666666-6666-4666-8666-666666666666"
REPORT = {
    "schema_version": "harbor_import_report.v5",
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


@pytest.fixture
def importer(platform: Platform) -> Platform:
    """The platform, with each import it starts completing and leaving ``REPORT``."""
    platform.next_import.report = REPORT
    return platform


def started(platform: Platform) -> Import:
    (only,) = platform.imports.values()
    return only


def bundle(platform: Platform) -> zipfile.ZipFile:
    uploaded = started(platform).bundle
    assert uploaded is not None
    return zipfile.ZipFile(io.BytesIO(uploaded))


@pytest.fixture
def folders(tmp_path: Path) -> list[str]:
    """A tasks folder and a job folder with one trial, as Harbor writes them."""
    write_files(
        tmp_path,
        {
            "tasks/hello/task.toml": "version = '1.0'\n",
            "tasks/hello/tests/test.sh": "exit 0\n",
            "run-1/config.json": "{}",
            "run-1/result.json": "{}",
            "run-1/hello__a/config.json": "{}",
            "run-1/hello__a/result.json": "{}",
        },
    )
    return [str(tmp_path / "tasks"), str(tmp_path / "run-1")]


def test_import_uploads_the_folders_and_imports_them_into_a_new_taskset(
    hud: Hud, importer: Platform, folders: list[str]
) -> None:
    result = hud("harbor", "import", *folders, "--taskset", "TB2 Sample", "--yes", "--json")

    assert result.exit_code == 0, result
    taskset = importer.taskset("tb2-sample")
    assert started(importer).taskset_id == taskset["id"]
    assert sorted(bundle(importer).namelist()) == [
        "run-1/config.json",
        "run-1/hello__a/config.json",
        "run-1/hello__a/result.json",
        "run-1/result.json",
        "tasks/hello/task.toml",
        "tasks/hello/tests/test.sh",
    ]
    assert "signature" not in str(result)


def test_rerunning_an_import_reuses_the_taskset_its_name_created(
    hud: Hud, importer: Platform, folders: list[str]
) -> None:
    first = hud("harbor", "import", *folders, "--taskset", "TB2/Sample", "--yes", "--json")
    second = hud("harbor", "import", *folders, "--taskset", "TB2/Sample", "--yes", "--json")

    assert (first.exit_code, second.exit_code) == (0, 0), second
    (taskset,) = importer.tasksets.values()
    assert second.json["taskset"] == {"id": taskset["id"], "name": "tb2sample", "created": False}


def test_import_link_saves_the_taskset_as_the_directory_default(
    hud: Hud, importer: Platform, folders: list[str]
) -> None:
    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--link")

    assert result.exit_code == 0, result
    linked = json.loads((hud.cwd / ".hud" / "config.json").read_text())["taskset_id"]
    assert linked == importer.taskset("tb2")["id"]


def test_import_uploads_a_given_zip_as_it_is(hud: Hud, importer: Platform, tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.zip"
    with zipfile.ZipFile(prepared, "w") as archive:
        archive.writestr("hello/task.toml", "version = '1.0'\n")

    result = hud("harbor", "import", str(prepared), "--yes", "--json")

    assert result.exit_code == 0, result
    assert started(importer).bundle == prepared.read_bytes()
    assert started(importer).taskset_id == importer.taskset("prepared")["id"]


def test_a_dry_run_plans_without_writing(hud: Hud, importer: Platform, folders: list[str]) -> None:
    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert result.json["taskset"]["created"] is True
    assert (importer.tasksets, importer.imports) == ({}, {})


def test_import_refuses_a_taskset_the_caller_cannot_edit(
    hud: Hud, importer: Platform, folders: list[str]
) -> None:
    theirs = importer.add_taskset("theirs", can_edit=False)

    result = hud("harbor", "import", *folders, "--taskset", theirs, "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "permission_denied"), result
    assert importer.imports == {}


def test_import_refuses_a_bundle_over_the_platform_limit(
    hud: Hud, importer: Platform, folders: list[str]
) -> None:
    importer.import_formats[0]["max_input_bytes"] = 10

    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result
    assert (importer.tasksets, importer.imports) == ({}, {})


def test_a_refused_upload_deletes_its_reservation_and_imports_nothing(
    hud: Hud, importer: Platform, folders: list[str]
) -> None:
    importer.services.route("api", "PUT", "/uploads/{id}", status=403)

    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--json")

    assert result.exit_code == 1, result
    assert importer.imports == {}
    assert "signature" not in str(result)


@pytest.mark.parametrize(
    ("statuses", "report", "exit_code"),
    [
        pytest.param(["queued", "running", "completed"], REPORT, 0, id="completed"),
        pytest.param(
            ["stopped"], {**REPORT, "error": {"step": "add", "message": "boom"}}, 1, id="stopped"
        ),
        pytest.param(["failed"], None, 1, id="failed"),
    ],
)
def test_import_waits_for_the_import_and_fails_unless_it_completed(
    hud: Hud,
    importer: Platform,
    folders: list[str],
    statuses: list[str],
    report: dict[str, Any] | None,
    exit_code: int,
) -> None:
    importer.next_import = Import(
        "",
        taskset_id="",
        statuses=statuses,
        report=report,
        error="crashed" if report is None else None,
    )

    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--json")

    assert result.exit_code == exit_code, result
    shown = result.json["report"]
    assert result.json["import"]["status"] == statuses[-1]
    assert (shown is None, (shown or {}).get("error")) == (
        report is None,
        (report or {}).get("error"),
    )


def test_no_wait_returns_the_started_import(
    hud: Hud, importer: Platform, folders: list[str]
) -> None:
    importer.next_import.statuses = ["queued", "completed"]

    result = hud("harbor", "import", *folders, "--taskset", "tb2", "--yes", "--no-wait", "--json")

    assert (result.exit_code, result.json["import"]["status"]) == (0, "queued"), result
    assert started(importer).statuses == ["completed"]


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
    platform: Platform,
    argv: list[str],
    statuses: list[str],
    report: dict[str, Any],
    version: str | None,
) -> None:
    platform.imports[IMPORT] = Import(IMPORT, "tb2", statuses=statuses, report=report)

    result = hud("harbor", "report", IMPORT, *argv, "--json")

    assert result.exit_code == 0, result
    shown_version = (result.json["report"] or {}).get("schema_version")
    assert (result.json["import"]["status"], shown_version) == (statuses[-1], version)


def test_report_refuses_an_import_of_another_format(hud: Hud, platform: Platform) -> None:
    platform.imports[IMPORT] = Import(IMPORT, "tb2", format="parquet")

    result = hud("harbor", "report", IMPORT, "--json")

    assert (result.exit_code, result.json["error"]) == (2, "usage"), result


@pytest.mark.parametrize(("status", "cancelled"), [("running", True), ("completed", False)])
def test_cancel_stops_only_a_live_import(
    hud: Hud, platform: Platform, status: str, cancelled: bool
) -> None:
    platform.imports[IMPORT] = Import(IMPORT, "tb2", statuses=[status])

    result = hud("harbor", "cancel", IMPORT, "--yes", "--json")

    assert result.exit_code == 0, result
    assert platform.imports[IMPORT].cancelled is cancelled


def test_imports_asks_for_a_tasksets_harbor_imports(hud: Hud, platform: Platform) -> None:
    tb2 = platform.add_taskset("tb2")
    platform.imports[IMPORT] = Import(IMPORT, tb2)

    result = hud("harbor", "imports", "--taskset", "tb2", "--json")

    assert result.exit_code == 0, result
    (listed,) = platform.services.requests("api", "GET", "/v2/imports")
    assert (listed.query["taskset_id"], listed.query["format"]) == ([tb2], ["harbor"])


def test_imports_of_an_unknown_taskset_is_not_found(hud: Hud, platform: Platform) -> None:
    result = hud("harbor", "imports", "--taskset", "nope", "--json")

    assert (result.exit_code, result.json["error"]) == (1, "not_found"), result


def harbor_task(root: Path) -> Path:
    """A minimal Harbor task folder, ``hello``, under ``root``."""
    files = {"task.toml": "version = '1.0'\n", "environment/Dockerfile": "FROM ubuntu\n"}
    write_files(root / "hello", files)
    return root / "hello"


def test_import_bundles_linked_files_by_content_and_skips_linked_folders(
    hud: Hud, importer: Platform, tmp_path: Path
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
    sent = bundle(importer)
    assert sent.read("hello/environment/data.txt") == b"shared"
    assert not any("linked" in name or "dangling" in name for name in sent.namelist())


def test_import_refuses_two_folders_with_one_name_before_writing(
    hud: Hud, importer: Platform, tmp_path: Path
) -> None:
    first, second = harbor_task(tmp_path / "a"), harbor_task(tmp_path / "b")

    result = hud("harbor", "import", str(first), str(second), "--yes")

    assert result.exit_code != 0, result
    assert (importer.tasksets, importer.imports) == ({}, {})
