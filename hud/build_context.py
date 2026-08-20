"""Canonical filesystem identity for HUD build contexts."""

from __future__ import annotations

import hashlib
import json
import stat
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


BUILD_CONTEXT_MANIFEST_VERSION = 1


class BuildContextEntry(BaseModel):
    """One build-semantic filesystem entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    type: Literal["file", "symlink"]
    mode: int = Field(ge=0, le=0o7777)
    size: int | None = Field(default=None, ge=0)
    content_digest: str | None = None
    target: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        if (
            self.path.startswith(("/", "\\"))
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
        ):
            raise ValueError(f"Build context path must be relative and normalized: {self.path!r}")
        if self.type == "file":
            if self.size is None or self.content_digest is None or self.target is not None:
                raise ValueError("File entries require size and content_digest only")
            if len(self.content_digest) != 64 or any(
                character not in "0123456789abcdef" for character in self.content_digest
            ):
                raise ValueError("File content_digest must be a lowercase SHA-256 digest")
        elif self.target is None or self.size is not None or self.content_digest is not None:
            raise ValueError("Symlink entries require target only")
        return self


class BuildContextManifest(BaseModel):
    """Canonical manifest shared by build-context archiving and identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = BUILD_CONTEXT_MANIFEST_VERSION
    entries: tuple[BuildContextEntry, ...]

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        paths = [entry.path for entry in self.entries]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("Build context entries must have unique paths in sorted order")
        return self

    @classmethod
    def from_paths(cls, root: Path, paths: Iterable[Path]) -> Self:
        resolved_root = root.resolve()
        entries = tuple(
            _entry_from_path(resolved_root, path)
            for path in sorted(paths, key=lambda path: path.relative_to(resolved_root).as_posix())
        )
        return cls(entries=entries)

    @classmethod
    def from_directory(cls, root: Path) -> Self:
        resolved_root = root.resolve()
        return cls.from_paths(
            resolved_root,
            (path for path in resolved_root.rglob("*") if path.is_symlink() or path.is_file()),
        )

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _entry_from_path(root: Path, path: Path) -> BuildContextEntry:
    relative_path = path.relative_to(root).as_posix()
    metadata = path.lstat()
    mode = stat.S_IMODE(metadata.st_mode)
    if path.is_symlink():
        return BuildContextEntry(
            path=relative_path,
            type="symlink",
            mode=mode,
            target=path.readlink().as_posix(),
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Unsupported build context entry: {relative_path}")

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return BuildContextEntry(
        path=relative_path,
        type="file",
        mode=mode,
        size=metadata.st_size,
        content_digest=digest.hexdigest(),
    )
