"""HUD Data: files in the team data store and the data pipelines that run on them."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from hud.utils.exceptions import HudException, HudNetworkError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path
    from typing import BinaryIO

    from hud.utils.platform import PlatformClient

_TRANSFER_TIMEOUT = httpx.Timeout(30.0, read=600.0, write=600.0)
_CHUNK_BYTES = 1024 * 1024

# A run is ``queued`` until it ends; ``phase`` tells where a live run is.
RunStatus = Literal["queued", "completed", "blocked", "error", "cancelled"]
RunPhase = Literal[
    "queued", "starting", "running", "cancelling", "completed", "blocked", "error", "cancelled"
]


class _Record(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class DataFile(_Record):
    id: str
    filename: str
    size_bytes: int | None = None


class DataPipeline(_Record):
    key: str
    title: str
    description: str | None = None
    max_input_bytes: int


class PipelineRunOutput(_Record):
    name: str
    file_id: str
    filename: str


class PipelineRun(_Record):
    id: str
    pipeline_key: str
    input_file_id: str
    input_filename: str
    input_size_bytes: int
    args: dict[str, Any]
    status: RunStatus
    phase: RunPhase
    # The pipeline's ``data_pipeline_result.v1`` envelope once it delivers.
    result: dict[str, Any] | None = None
    error: str | None = None
    outputs: list[PipelineRunOutput] = Field(default_factory=list[PipelineRunOutput])
    created_at: datetime
    completed_at: datetime | None = None
    credits_used: float | None = None

    @property
    def finished(self) -> bool:
        return self.status != "queued"

    def output(self, name: str) -> PipelineRunOutput | None:
        return next((output for output in self.outputs if output.name == name), None)


def upload_file(
    platform: PlatformClient,
    path: Path,
    *,
    content_type: str,
    pipeline_key: str | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> DataFile:
    """Store a local file in the caller's data store and return it once ready.

    ``pipeline_key`` reserves the file as that pipeline's input, held to the
    pipeline's ``max_input_bytes`` instead of the per-file limit. A reservation
    whose upload fails is deleted. ``on_progress`` receives each sent chunk's size.
    """
    size = path.stat().st_size
    reserved = platform.post(
        "/data",
        json={
            "filename": path.name,
            "size_bytes": size,
            "content_type": content_type,
            "scope": "member",
            "pipeline_key": pipeline_key,
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
        _check_storage(response, "upload")
        return DataFile.model_validate(platform.post(f"/data/{file_id}/complete"))
    except BaseException as exc:
        try:
            platform.delete(f"/data/{file_id}")
        except HudException as cleanup:
            exc.add_note(f"The reserved data file {file_id} could not be deleted: {cleanup}")
        raise


def download_json(platform: PlatformClient, file_id: str) -> Any:
    """The decoded contents of a JSON file in the data store."""
    links = platform.get(f"/data/{file_id}/download")
    with httpx.Client(timeout=_TRANSFER_TIMEOUT) as client:
        response = client.get(links["download_url"])
    _check_storage(response, "download")
    return response.json()


def list_pipelines(platform: PlatformClient) -> list[DataPipeline]:
    """The pipelines this platform offers."""
    data = platform.get("/data-pipelines")
    return [DataPipeline.model_validate(item) for item in data["pipelines"]]


def start_run(
    platform: PlatformClient,
    pipeline_key: str,
    file_id: str,
    args: Mapping[str, Any],
) -> PipelineRun:
    """Run a pipeline on a ready data file with the pipeline's own run arguments."""
    record = platform.post(
        "/data-pipelines/runs",
        json={"pipeline_key": pipeline_key, "file_id": file_id, "args": dict(args)},
    )
    return PipelineRun.model_validate(record)


def get_run(platform: PlatformClient, run_id: str) -> PipelineRun:
    return PipelineRun.model_validate(platform.get(f"/data-pipelines/runs/{run_id}"))


def list_runs(
    platform: PlatformClient,
    *,
    pipeline_key: str,
    args: Mapping[str, str] | None = None,
    limit: int = 50,
) -> tuple[list[PipelineRun], int]:
    """The caller's runs of a pipeline, newest first, and how many there are in total.

    ``args`` keeps runs whose run argument of that name has that value.
    """
    params: dict[str, Any] = {"pipeline_key": pipeline_key, "limit": limit}
    if args:
        params["arg"] = [f"{name}:{value}" for name, value in args.items()]
    data = platform.get("/data-pipelines/runs", params=params)
    return [PipelineRun.model_validate(item) for item in data["runs"]], int(data["total"])


def cancel_run(platform: PlatformClient, run_id: str) -> PipelineRun:
    """Request cancellation of a run; a run that has ended is returned unchanged."""
    return PipelineRun.model_validate(platform.post(f"/data-pipelines/runs/{run_id}/cancel"))


def wait_for_run(
    platform: PlatformClient,
    run_id: str,
    *,
    poll_seconds: float = 3.0,
    on_update: Callable[[PipelineRun], None] | None = None,
) -> PipelineRun:
    """Poll a run until it ends; ``on_update`` sees every poll."""
    while True:
        run = get_run(platform, run_id)
        if on_update is not None:
            on_update(run)
        if run.finished:
            return run
        time.sleep(poll_seconds)


def _chunks(stream: BinaryIO, on_progress: Callable[[int], None] | None) -> Iterator[bytes]:
    while chunk := stream.read(_CHUNK_BYTES):
        yield chunk
        if on_progress is not None:
            on_progress(len(chunk))


def _check_storage(response: httpx.Response, action: str) -> None:
    # The URL is presigned, so the error names only the status.
    if response.is_error:
        raise HudNetworkError(f"Storage refused the {action}: HTTP {response.status_code}")
