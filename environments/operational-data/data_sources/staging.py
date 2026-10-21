"""Downloading uploaded data files with the HUD credentials the rollout provides."""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Any, Self

import httpx
from hud.environment import DataFileRef
from hud.settings import settings

_TIMEOUT = httpx.Timeout(30.0, read=300.0)
_CHUNK = 1 << 20
SAMPLES_ROOT = Path(__file__).resolve().parent.parent / "samples"


class DataFileError(RuntimeError):
    """A referenced data file could not be downloaded."""


class UnsafePath(ValueError):
    """A relative path that is empty or would escape its root."""


def destination(root: Path, relative: str) -> Path:
    """``relative`` under ``root``, refusing paths that would escape it."""
    candidate = PurePosixPath(relative)
    if (
        not candidate.parts
        or candidate.is_absolute()
        or ".." in candidate.parts
        or "\\" in relative
    ):
        raise UnsafePath(f"unsafe path: {relative!r}")
    return root / candidate


def copy_bundled(relative: str, root: Path) -> Path:
    """Stage one bundled sample under its basename, refusing links and collisions."""
    source = destination(SAMPLES_ROOT, relative)
    parts = PurePosixPath(relative).parts
    if any(SAMPLES_ROOT.joinpath(*parts[:index]).is_symlink() for index in range(len(parts) + 1)):
        raise UnsafePath(f"bundled sample is a symlink: {relative!r}")
    if not source.is_file():
        raise DataFileError(f"bundled sample does not exist: {relative!r}")
    target = destination(root, source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = target.open("xb")
    except FileExistsError as error:
        raise DataFileError(f"two data files would be saved as {source.name!r}") from error
    with out, source.open("rb") as incoming:
        shutil.copyfileobj(incoming, out)
    return target


class DataFiles:
    """A download session against the HUD data-file API."""

    def __init__(self) -> None:
        key = (settings.api_key or "").strip()
        if not key:
            raise DataFileError("HUD_API_KEY is unset; the environment cannot read data files")
        self._base = settings.hud_api_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._client = httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    async def _json(self, path: str) -> dict[str, Any]:
        response = await self._client.get(f"{self._base}{path}", headers=self._headers)
        if response.is_error:
            raise DataFileError(f"GET {path} failed: {response.status_code}")
        return response.json()

    async def download(self, ref: DataFileRef, root: Path) -> Path:
        """Download a file into ``root`` at ``ref.path``, or under its uploaded name.

        Two files resolving to the same destination fail rather than overwrite each other.
        """
        metadata = await self._json(f"/v2/data/{ref.file_id}")
        filename = metadata.get("filename")
        if not isinstance(filename, str) or not filename:
            raise DataFileError(f"data file {ref.file_id} has no filename")
        name = ref.path or PurePosixPath(filename).name
        target = destination(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        url = (await self._json(f"/v2/data/{ref.file_id}/download")).get("url")
        if not isinstance(url, str) or not url:
            raise DataFileError(f"data file {ref.file_id} has no download url")
        try:
            out = target.open("xb")
        except FileExistsError as error:
            raise DataFileError(f"two data files would be saved as {name!r}") from error
        with out:
            # The presigned URL carries its own credentials; ours would be rejected.
            async with self._client.stream("GET", url) as response:
                if response.is_error:
                    raise DataFileError(
                        f"downloading data file {ref.file_id} failed: {response.status_code}"
                    )
                async for chunk in response.aiter_bytes(_CHUNK):
                    out.write(chunk)
        return target
