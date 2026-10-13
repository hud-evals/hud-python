"""HUD imports: upload a bundle and import it into a taskset."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from hud.utils.exceptions import HudException, HudNetworkError, HudRequestError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import BinaryIO

    from hud.utils.platform import PlatformClient

_TRANSFER_TIMEOUT = httpx.Timeout(30.0, read=600.0, write=600.0)
_CHUNK_BYTES = 1024 * 1024

ImportStatus = Literal[
    "queued",
    "starting",
    "running",
    "cancelling",
    "completed",
    "stopped",
    "failed",
    "cancelled",
]
# ``stopped``: the importer refused the bundle or stopped partway, and its report
# says why. ``failed``: the import did not run to an outcome.
LIVE_STATUSES: frozenset[ImportStatus] = frozenset({"queued", "starting", "running", "cancelling"})


class _Record(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class ImportFormat(_Record):
    format: str
    title: str
    description: str | None = None
    max_input_bytes: int


class ImportFile(_Record):
    id: str
    filename: str
    size_bytes: int


class Import(_Record):
    id: str
    format: str
    taskset_id: str
    file: ImportFile
    status: ImportStatus
    # Set once the import completed or stopped.
    summary: str | None = None
    metrics: dict[str, float] | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None

    @property
    def ended(self) -> bool:
        return self.status not in LIVE_STATUSES

    @property
    def publishes_report(self) -> bool:
        return self.status in {"completed", "stopped"}


def list_formats(platform: PlatformClient) -> list[ImportFormat]:
    """The bundle formats this platform imports."""
    data = platform.get("/imports/formats")
    return [ImportFormat.model_validate(item) for item in data["formats"]]


def upload_bundle(
    platform: PlatformClient,
    path: Path,
    *,
    format: str,
    content_type: str,
    on_progress: Callable[[int], None] | None = None,
) -> str:
    """Store a local bundle in the caller's data store and return its file id once ready.

    The bundle is held to the format's ``max_input_bytes``. A reservation whose
    upload fails is deleted. ``on_progress`` receives each sent chunk's size.
    """
    size = path.stat().st_size
    reserved = platform.post(
        "/imports/uploads",
        json={
            "format": format,
            "filename": path.name,
            "size_bytes": size,
            "content_type": content_type,
        },
    )
    file_id = str(reserved["file"]["id"])
    try:
        with path.open("rb") as stream, httpx.Client(timeout=_TRANSFER_TIMEOUT) as client:
            response = client.put(
                reserved["upload_url"],
                content=_chunks(stream, on_progress),
                # S3 refuses chunked uploads, so the length goes up front.
                headers={"Content-Type": content_type, "Content-Length": str(size)},
            )
        if response.is_error:
            # The URL is presigned, so the error names only the status.
            raise HudNetworkError(f"Storage refused the upload: HTTP {response.status_code}")
        platform.post(f"/data/{file_id}/complete")
    except BaseException as exc:
        try:
            platform.delete(f"/data/{file_id}")
        except HudException as cleanup:
            exc.add_note(f"The reserved data file {file_id} could not be deleted: {cleanup}")
        raise
    return file_id


def start_import(platform: PlatformClient, *, format: str, file_id: str, taskset_id: str) -> Import:
    """Import an uploaded bundle into a taskset the caller can edit."""
    record = platform.post(
        "/imports", json={"format": format, "file_id": file_id, "taskset_id": taskset_id}
    )
    return Import.model_validate(record)


def get_import(platform: PlatformClient, import_id: str) -> Import:
    return Import.model_validate(platform.get(f"/imports/{import_id}"))


def list_imports(
    platform: PlatformClient,
    *,
    format: str | None = None,
    taskset_id: str | None = None,
    limit: int = 50,
) -> tuple[list[Import], int]:
    """The caller's imports, newest first, and how many there are in total."""
    params: dict[str, Any] = {"limit": limit}
    if format is not None:
        params["format"] = format
    if taskset_id is not None:
        params["taskset_id"] = taskset_id
    data = platform.get("/imports", params=params)
    return [Import.model_validate(item) for item in data["imports"]], int(data["total"])


def cancel_import(platform: PlatformClient, import_id: str) -> Import:
    """Request cancellation of an import; one that has ended is returned unchanged."""
    return Import.model_validate(platform.post(f"/imports/{import_id}/cancel"))


def get_report(platform: PlatformClient, import_id: str) -> Any:
    """The report a completed or stopped import published, in its format's schema.

    None when the import published no report.
    """
    try:
        return platform.get(f"/imports/{import_id}/report")
    except HudRequestError as exc:
        if exc.status_code == 404:
            return None
        raise


def wait_for_import(
    platform: PlatformClient,
    import_id: str,
    *,
    poll_seconds: float = 3.0,
    on_update: Callable[[Import], None] | None = None,
) -> Import:
    """Poll an import until it ends; ``on_update`` sees every poll."""
    while True:
        imported = get_import(platform, import_id)
        if on_update is not None:
            on_update(imported)
        if imported.ended:
            return imported
        time.sleep(poll_seconds)


def _chunks(stream: BinaryIO, on_progress: Callable[[int], None] | None) -> Iterator[bytes]:
    while chunk := stream.read(_CHUNK_BYTES):
        yield chunk
        if on_progress is not None:
            on_progress(len(chunk))
