"""Import completed Harbor results into the HUD platform."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field

from hud.eval.sync import resolve_taskset_id
from hud.utils.exceptions import HudConfigError, HudNetworkError, HudRequestError, HudTimeoutError
from hud.utils.platform import PlatformClient

UPLOAD_TIMEOUT_SECONDS = 600.0


class _DataFileReference(BaseModel):
    id: UUID


class _DataFileReservation(BaseModel):
    file: _DataFileReference
    upload_url: str = Field(min_length=1)


class _HarborImportSubmission(BaseModel):
    job_id: UUID
    trace_id: UUID


class HarborImportResult(_HarborImportSubmission):
    """Identifiers created when a completed Harbor archive is submitted."""

    model_config = ConfigDict(frozen=True)

    data_file_id: UUID


def _archive_path(archive: str | Path | None) -> Path:
    if archive is None:
        raise HudConfigError("A Harbor archive or --data-file-id is required")
    path = Path(archive).expanduser().resolve()
    if not path.is_file():
        raise HudConfigError(f"Harbor archive is not a file: {path}")
    if path.stat().st_size == 0:
        raise HudConfigError(f"Harbor archive is empty: {path}")
    return path


def _archive_content_type(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".zip"):
        return "application/zip"
    if lower_name.endswith((".tar.gz", ".tgz", ".gz")):
        return "application/gzip"
    return "application/octet-stream"


def _upload_archive(upload_url: str, archive: Path, content_type: str) -> None:
    headers = {
        "Content-Length": str(archive.stat().st_size),
        "Content-Type": content_type,
    }
    try:
        with archive.open("rb") as source, httpx.Client(timeout=UPLOAD_TIMEOUT_SECONDS) as client:
            response = client.put(upload_url, content=source, headers=headers)
            response.raise_for_status()
    except httpx.TimeoutException as error:
        raise HudTimeoutError(f"Harbor archive upload timed out: {error}") from None
    except httpx.HTTPStatusError as error:
        raise HudRequestError.from_httpx_error(
            error,
            context="Harbor archive upload failed",
        ) from None
    except httpx.RequestError as error:
        raise HudNetworkError(f"Harbor archive upload failed: {error}") from None


def _resolve_taskset(
    platform: PlatformClient,
    taskset: str | UUID | None,
) -> UUID | None:
    if taskset is None or isinstance(taskset, UUID):
        return taskset
    taskset_id, _ = resolve_taskset_id(platform, taskset)
    if not taskset_id:
        raise HudRequestError(f"Taskset not found: {taskset}", status_code=404)
    return UUID(taskset_id)


def import_results(
    archive: str | Path | None = None,
    *,
    taskset: str | UUID | None = None,
    data_file_id: UUID | None = None,
    platform: PlatformClient | None = None,
) -> HarborImportResult:
    """Upload a completed Harbor archive and submit it for asynchronous import.

    Pass ``data_file_id`` to retry submission without uploading the archive again.
    Supplying ``taskset`` associates imported trials with the taskset's current
    task versions.
    """
    client = platform or PlatformClient.from_settings()
    taskset_id = _resolve_taskset(client, taskset)

    if data_file_id is None:
        archive_path = _archive_path(archive)
        content_type = _archive_content_type(archive_path)
        reservation = _DataFileReservation.model_validate(
            client.post(
                "/data",
                json={
                    "filename": archive_path.name,
                    "size_bytes": archive_path.stat().st_size,
                    "content_type": content_type,
                    "scope": "member",
                },
            )
        )
        _upload_archive(reservation.upload_url, archive_path, content_type)
        client.post(f"/data/{reservation.file.id}/complete")
        data_file_id = reservation.file.id

    submission = _HarborImportSubmission.model_validate(
        client.post(
            "/jobs/import/harbor",
            json={
                "data_file_id": str(data_file_id),
                "taskset_id": str(taskset_id) if taskset_id is not None else None,
            },
        )
    )
    return HarborImportResult(data_file_id=data_file_id, **submission.model_dump())


__all__ = ["HarborImportResult", "import_results"]
