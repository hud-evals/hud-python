"""Completed Harbor result import."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx
import pytest

from hud.integrations.harbor import import_results
from hud.utils.exceptions import HudConfigError, HudRequestError
from hud.utils.platform import PlatformClient

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

DATA_FILE_ID = UUID("00000000-0000-4000-a000-000000000001")
TASKSET_ID = UUID("00000000-0000-4000-a000-000000000002")
JOB_ID = UUID("00000000-0000-4000-a000-000000000003")
TRACE_ID = UUID("00000000-0000-4000-a000-000000000004")


def _platform(monkeypatch: pytest.MonkeyPatch) -> tuple[PlatformClient, list[tuple[str, Any]]]:
    posts: list[tuple[str, Any]] = []

    def post(_self: PlatformClient, path: str, *, json: Any | None = None) -> Any:
        posts.append((path, json))
        if path == "/data":
            return {
                "file": {"id": str(DATA_FILE_ID)},
                "upload_url": "https://uploads.example/harbor",
            }
        if path == "/jobs/import/harbor":
            return {"job_id": str(JOB_ID), "trace_id": str(TRACE_ID)}
        return {}

    monkeypatch.setattr(PlatformClient, "post", post)
    return PlatformClient("https://api.example", "key"), posts


def _mock_upload(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    client_type = httpx.Client
    transport = httpx.MockTransport(handler)
    module = import_module("hud.integrations.harbor.import_results")
    monkeypatch.setattr(
        module.httpx, "Client", lambda **kwargs: client_type(transport=transport, **kwargs)
    )


def test_import_results_uploads_archive_and_submits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "harbor-job.zip"
    archive.write_bytes(b"harbor-results")
    platform, posts = _platform(monkeypatch)
    uploads: list[tuple[bytes, str, str]] = []

    def upload(request: httpx.Request) -> httpx.Response:
        uploads.append(
            (
                request.read(),
                request.headers["content-length"],
                request.headers["content-type"],
            )
        )
        return httpx.Response(200)

    _mock_upload(monkeypatch, upload)

    result = import_results(archive, taskset=TASKSET_ID, platform=platform)

    assert result.data_file_id == DATA_FILE_ID
    assert result.job_id == JOB_ID
    assert result.trace_id == TRACE_ID
    assert uploads == [(b"harbor-results", "14", "application/zip")]
    assert posts == [
        (
            "/data",
            {
                "filename": "harbor-job.zip",
                "size_bytes": 14,
                "content_type": "application/zip",
                "scope": "member",
            },
        ),
        (f"/data/{DATA_FILE_ID}/complete", None),
        (
            "/jobs/import/harbor",
            {"data_file_id": str(DATA_FILE_ID), "taskset_id": str(TASKSET_ID)},
        ),
    ]


def test_import_results_reuses_data_file_without_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform, posts = _platform(monkeypatch)

    result = import_results(data_file_id=DATA_FILE_ID, platform=platform)

    assert result.data_file_id == DATA_FILE_ID
    assert posts == [
        (
            "/jobs/import/harbor",
            {"data_file_id": str(DATA_FILE_ID), "taskset_id": None},
        )
    ]


def test_import_results_resolves_taskset_name(monkeypatch: pytest.MonkeyPatch) -> None:
    platform, posts = _platform(monkeypatch)

    def get(_self: PlatformClient, path: str, *, params: dict[str, Any] | None = None) -> Any:
        assert path == "/tasksets/by-name/terminal-bench"
        assert params is None
        return {"taskset_id": str(TASKSET_ID), "name": "terminal-bench"}

    monkeypatch.setattr(PlatformClient, "get", get)

    import_results(data_file_id=DATA_FILE_ID, taskset="terminal-bench", platform=platform)

    assert posts[-1] == (
        "/jobs/import/harbor",
        {"data_file_id": str(DATA_FILE_ID), "taskset_id": str(TASKSET_ID)},
    )


def test_import_results_refuses_unknown_taskset(monkeypatch: pytest.MonkeyPatch) -> None:
    platform, posts = _platform(monkeypatch)

    def get(_self: PlatformClient, path: str, *, params: dict[str, Any] | None = None) -> Any:
        raise HudRequestError("missing", status_code=404)

    monkeypatch.setattr(PlatformClient, "get", get)

    with pytest.raises(HudRequestError, match="Taskset not found: missing"):
        import_results(data_file_id=DATA_FILE_ID, taskset="missing", platform=platform)

    assert posts == []


@pytest.mark.parametrize("contents", [None, b""])
def test_import_results_requires_nonempty_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    contents: bytes | None,
) -> None:
    platform, posts = _platform(monkeypatch)
    archive = tmp_path / "harbor.zip"
    if contents is not None:
        archive.write_bytes(contents)

    with pytest.raises(HudConfigError):
        import_results(archive, platform=platform)

    assert posts == []


def test_import_results_wraps_upload_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "harbor.tar.gz"
    archive.write_bytes(b"archive")
    platform, posts = _platform(monkeypatch)
    _mock_upload(monkeypatch, lambda _request: httpx.Response(403, text="expired"))

    with pytest.raises(HudRequestError, match="Harbor archive upload failed"):
        import_results(archive, platform=platform)

    assert [path for path, _ in posts] == ["/data"]
