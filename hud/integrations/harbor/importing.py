"""Packing Harbor folders for the ``harbor_import`` pipeline and reading its report."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Sequence

PIPELINE_KEY = "harbor_import"
REPORT_OUTPUT = "report"
REPORT_SCHEMA = "harbor_import_report.v4"

# Files only a Harbor task folder (task.toml) or job and trial folder (result.json) holds.
_HARBOR_FILES = frozenset({"task.toml", "result.json"})

ItemStatus = Literal["imported", "unchanged", "skipped", "failed"]


class BundleError(ValueError):
    """The folders cannot be packed into a Harbor bundle."""


def pack_bundle(paths: Sequence[Path], destination: Path) -> int:
    """Zip each folder under its own name into ``destination``; returns how many files it holds.

    Files go in unchanged so the import computes Harbor's task digests. Like
    Harbor's packager, symlinked files are stored by content and symlinked
    folders are skipped.
    """
    folders: dict[str, Path] = {}
    for path in paths:
        if not path.is_dir():
            raise BundleError(f"{path} is not a folder")
        name = path.resolve().name
        if not name:
            raise BundleError(f"{path} has no folder name to store it under")
        if name in folders:
            raise BundleError(
                f"{folders[name]} and {path} are both named {name!r}; "
                "a bundle holds one folder of each name"
            )
        folders[name] = path

    count = 0
    holds_harbor_files = False
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in folders.items():
            root = path.resolve()
            for current, dirnames, filenames in os.walk(root):
                dirnames.sort()
                directory = Path(current)
                for filename in sorted(filenames):
                    file = directory / filename
                    if not file.is_file():
                        continue
                    member = PurePosixPath(name, file.relative_to(root).as_posix())
                    archive.write(file, member.as_posix())
                    count += 1
                    holds_harbor_files = holds_harbor_files or filename in _HARBOR_FILES
    if not holds_harbor_files:
        raise BundleError(
            "The folders hold no Harbor task folder (a folder with task.toml) "
            "and no Harbor job folder (a folder with result.json)"
        )
    return count


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class ReportTask(_Item):
    """A task folder in the bundle, or a task in HUD that one of its trials ran."""

    name: str
    source: Literal["bundle", "hud"]
    path: str | None = None
    environment_id: str | None = None
    build_id: str | None = None
    build_version: int | None = None
    task_id: str | None = None
    status: ItemStatus
    reason: str | None = None


class ReportTrial(_Item):
    path: str
    status: ItemStatus
    name: str | None = None
    task: str | None = None
    trace_id: str | None = None
    reason: str | None = None


class ReportJob(_Item):
    path: str
    name: str | None = None
    job_id: str | None = None
    job_status: str | None = None
    status: ItemStatus
    reason: str | None = None
    trials: list[ReportTrial]


class ReportProblem(_Item):
    path: str
    reason: str


class ReportError(_Item):
    step: str
    message: str


class ImportReport(_Item):
    """The ``report`` output of a ``harbor_import`` run."""

    schema_version: Literal["harbor_import_report.v4"]
    taskset_id: str
    tasks: list[ReportTask]
    jobs: list[ReportJob]
    warnings: list[ReportProblem]
    # Set when the import stopped, with the step it stopped at.
    error: ReportError | None = None
