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
    from collections.abc import Callable
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
FOLDERS = {
    "tasks/hello/task.toml": "version = '1.0'\n",
    "tasks/hello/tests/test.sh": "exit 0\n",
    "run-1/config.json": "{}",
    "run-1/result.json": "{}",
    "run-1/hello__a/config.json": "{}",
    "run-1/hello__a/result.json": "{}",
}


@pytest.fixture
def importer(platform: Platform) -> Platform:
    """The platform, with each import it starts completing and leaving ``REPORT``."""
    platform.next_import.report = REPORT
    return platform


def started(platform: Platform) -> Import:
    (only,) = platform.imports.values()
    return only


def bundled(platform: Platform) -> dict[str, bytes]:
    """The bundle the platform received, member by member."""
    uploaded = started(platform).bundle
    assert uploaded is not None
    with zipfile.ZipFile(io.BytesIO(uploaded)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def harbor_folders(root: Path) -> list[str]:
    write_files(root, FOLDERS)
    return [str(root / "tasks"), str(root / "run-1")]


def a_prepared_zip(root: Path) -> list[str]:
    prepared = root / "prepared.zip"
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(prepared, "w") as archive:
        archive.writestr("hello/task.toml", "version = '1.0'\n")
    return [str(prepared)]


def a_task_with_links(root: Path) -> list[str]:
    task = root / "src" / "hello"
    write_files(task, {"task.toml": "version = '1.0'\n", "environment/Dockerfile": "FROM ubuntu\n"})
    write_files(root / "shared", {"data.txt": "shared"})
    (task / "environment" / "data.txt").symlink_to(root / "shared" / "data.txt")
    (task / "environment" / "linked").symlink_to(root / "shared")
    (task / "dangling").symlink_to(root / "missing")
    return [str(task)]


BUNDLES: dict[str, tuple[Callable[[Path], list[str]], dict[str, bytes], str]] = {
    "folders are zipped by their own paths": (
        harbor_folders,
        {name: content.encode() for name, content in FOLDERS.items()},
        "tb2",
    ),
    "a prepared zip is sent as it is": (
        a_prepared_zip,
        {"hello/task.toml": b"version = '1.0'\n"},
        "prepared",
    ),
    "a linked file is sent by content and a linked folder is skipped": (
        a_task_with_links,
        {
            "hello/task.toml": b"version = '1.0'\n",
            "hello/environment/Dockerfile": b"FROM ubuntu\n",
            "hello/environment/data.txt": b"shared",
        },
        "tb2",
    ),
}


@pytest.mark.parametrize(("source", "members", "taskset"), BUNDLES.values(), ids=BUNDLES.keys())
def test_an_import_uploads_its_input_as_one_bundle(
    hud: Hud,
    importer: Platform,
    tmp_path: Path,
    source: Callable[[Path], list[str]],
    members: dict[str, bytes],
    taskset: str,
) -> None:
    paths = source(tmp_path / "input")
    named = ["--taskset", taskset] if taskset != "prepared" else []

    result = hud("harbor", "import", *paths, *named, "--yes", "--json")

    assert result.exit_code == 0, result
    assert bundled(importer) == members
    assert started(importer).taskset_id == importer.taskset(taskset)["id"]


@pytest.mark.parametrize(
    ("runs", "flags", "created", "linked"),
    [
        pytest.param(1, [], True, False, id="a-new-name-creates-its-taskset"),
        pytest.param(2, [], False, False, id="rerunning-reuses-the-taskset-its-name-made"),
        pytest.param(1, ["--link"], True, True, id="link-saves-it-as-the-directory-default"),
        pytest.param(1, ["--dry-run"], True, False, id="a-dry-run-plans-to-create-it"),
    ],
)
def test_an_import_lands_in_the_taskset_its_name_resolves_to(
    hud: Hud,
    importer: Platform,
    tmp_path: Path,
    runs: int,
    flags: list[str],
    created: bool,
    linked: bool,
) -> None:
    paths = harbor_folders(tmp_path / "input")
    confirm = [] if "--dry-run" in flags else ["--yes"]

    results = [
        hud("harbor", "import", *paths, "--taskset", "TB2/Sample", *flags, *confirm, "--json")
        for _ in range(runs)
    ]

    assert [result.exit_code for result in results] == [0] * runs, results[-1]
    assert results[-1].json["taskset"]["created"] is created
    link = hud.cwd / ".hud" / "config.json"
    stored = json.loads(link.read_text())["taskset_id"] if link.exists() else None
    if "--dry-run" in flags:
        assert importer.tasksets == {}
    else:
        assert results[-1].json["taskset"] == {
            "id": importer.taskset("tb2sample")["id"],
            "name": "tb2sample",
            "created": created,
        }
        assert stored == (importer.taskset("tb2sample")["id"] if linked else None)


def test_a_refused_upload_deletes_its_reservation_and_imports_nothing(
    hud: Hud, importer: Platform, tmp_path: Path
) -> None:
    importer.services.route("api", "PUT", "/uploads/{id}", status=403)

    result = hud(
        "harbor", "import", *harbor_folders(tmp_path), "--taskset", "tb2", "--yes", "--json"
    )

    assert result.exit_code == 1, result
    assert importer.imports == {}


@pytest.mark.parametrize(
    ("statuses", "report", "flags", "exit_code", "shown"),
    [
        pytest.param(
            ["queued", "running", "completed"], REPORT, [], 0, "completed", id="completed"
        ),
        pytest.param(
            ["stopped"],
            {**REPORT, "error": {"step": "add", "message": "boom"}},
            [],
            1,
            "stopped",
            id="stopped",
        ),
        pytest.param(["failed"], None, [], 1, "failed", id="failed"),
        pytest.param(
            ["queued", "completed"], REPORT, ["--no-wait"], 0, "queued", id="no-wait-returns-it"
        ),
    ],
)
def test_an_import_waits_for_its_end_and_fails_unless_it_completed(
    hud: Hud,
    importer: Platform,
    tmp_path: Path,
    statuses: list[str],
    report: dict[str, Any] | None,
    flags: list[str],
    exit_code: int,
    shown: str,
) -> None:
    importer.next_import = Import(
        "",
        taskset_id="",
        statuses=statuses,
        report=report,
        error="crashed" if report is None else None,
    )
    paths = harbor_folders(tmp_path)

    result = hud("harbor", "import", *paths, "--taskset", "tb2", *flags, "--yes", "--json")

    assert (result.exit_code, result.json["import"]["status"]) == (exit_code, shown), result
    if not flags:
        reported = result.json["report"]
        assert (reported or {}).get("error") == (report or {}).get("error")


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
