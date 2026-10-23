"""HUD imports: upload a bundle and import it into a taskset."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import httpx
from pydantic import AliasPath, BaseModel, ConfigDict, Field

from hud.utils.exceptions import HudException, HudNetworkError, HudRequestError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import BinaryIO

    from hud.utils.platform import PlatformClient

T = TypeVar("T")

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


class _Formats(_Record):
    formats: list[ImportFormat]


class _Reservation(_Record):
    file_id: str = Field(validation_alias=AliasPath("file", "id"))
    upload_url: str


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


class _ImportPage(_Record):
    imports: list[Import]
    total: int


def list_formats(platform: PlatformClient) -> list[ImportFormat]:
    """The bundle formats this platform imports."""
    return platform.get("/imports/formats", _Formats).formats


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
        _Reservation,
        json={
            "format": format,
            "filename": path.name,
            "size_bytes": size,
            "content_type": content_type,
        },
    )
    file_id = reserved.file_id
    try:
        with path.open("rb") as stream, httpx.Client(timeout=_TRANSFER_TIMEOUT) as client:
            response = client.put(
                reserved.upload_url,
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
    return platform.post(
        "/imports",
        Import,
        json={"format": format, "file_id": file_id, "taskset_id": taskset_id},
    )


def get_import(platform: PlatformClient, import_id: str) -> Import:
    return platform.get(f"/imports/{import_id}", Import)


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
    page = platform.get("/imports", _ImportPage, params=params)
    return page.imports, page.total


def cancel_import(platform: PlatformClient, import_id: str) -> Import:
    """Request cancellation of an import; one that has ended is returned unchanged."""
    return platform.post(f"/imports/{import_id}/cancel", Import)


def get_report(platform: PlatformClient, import_id: str, reply: type[T]) -> T | None:
    """The report a completed or stopped import published, read as its format's ``reply``.

    None when the import published no report.
    """
    try:
        return platform.get(f"/imports/{import_id}/report", reply)
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
