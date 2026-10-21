"""Downloads from the HUD data-file API: safe destinations and the runtime credential."""

from functools import partial
from pathlib import Path

import httpx
import pytest
from hud.environment import DataFileRef

from data_sources import staging


def test_destinations_cannot_leave_the_root(tmp_path: Path):
    assert staging.destination(tmp_path, "notes/summary.md") == tmp_path / "notes/summary.md"
    for hostile in ("../escape.md", "/etc/passwd", "..\\escape.md", ""):
        with pytest.raises(staging.UnsafePath):
            staging.destination(tmp_path, hostile)


async def test_downloads_use_the_runtime_credential_and_uploaded_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(staging.settings, "api_key", "runtime-key")
    monkeypatch.setattr(staging.settings, "hud_api_url", "https://api.test")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            assert "Authorization" not in request.headers
            return httpx.Response(200, content=b"Summary,Issue key,Status\n")
        assert request.headers["Authorization"] == "Bearer runtime-key"
        if request.url.path.endswith("/download"):
            return httpx.Response(200, json={"url": "https://storage.test/blob"})
        return httpx.Response(200, json={"filename": "uploads/Jira export.csv"})

    monkeypatch.setattr(
        staging.httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)),
    )
    async with staging.DataFiles() as files:
        target = await files.download(DataFileRef(file_id="f1"), tmp_path)
    assert target == tmp_path / "Jira export.csv"
    assert target.read_bytes() == b"Summary,Issue key,Status\n"


async def test_a_ref_path_places_the_file_and_collisions_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(staging.settings, "api_key", "runtime-key")
    monkeypatch.setattr(staging.settings, "hud_api_url", "https://api.test")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "storage.test":
            return httpx.Response(200, content=request.url.path.encode())
        if request.url.path.endswith("/download"):
            return httpx.Response(200, json={"url": f"https://storage.test{request.url.path}"})
        return httpx.Response(200, json={"filename": "aging.xlsx"})

    monkeypatch.setattr(
        staging.httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)),
    )
    async with staging.DataFiles() as files:
        placed = await files.download(DataFileRef(file_id="f1", path="ap/march.xlsx"), tmp_path)
        assert placed == tmp_path / "ap" / "march.xlsx"
        await files.download(DataFileRef(file_id="f2"), tmp_path)
        with pytest.raises(staging.DataFileError, match="aging.xlsx"):
            await files.download(DataFileRef(file_id="f3"), tmp_path)
    assert (tmp_path / "aging.xlsx").read_bytes() == b"/v2/data/f2/download"


def test_a_missing_credential_is_reported_before_any_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(staging.settings, "api_key", None)
    with pytest.raises(staging.DataFileError, match="HUD_API_KEY"):
        staging.DataFiles()
