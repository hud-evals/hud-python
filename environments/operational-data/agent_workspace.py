"""The agent's workspace: emptied per task, attachments in files/, deliverables in outputs/."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from hud.environment import DataFileRef

from data_sources.staging import DataFiles, copy_bundled, destination
from documents import MAX_FILE_BYTES, UnreadableDocument, document_text

WORKSPACE_ROOT = Path("/workspace")
FILES = "files"
OUTPUTS = "outputs"

# The agent's shell runs as this uid when the server is root (the image's `agent` user),
# with the system Python first on PATH rather than the serving venv.
SHELL_UID = 1000
SHELL_PATH = "/usr/local/bin:/usr/bin:/bin"


def _hand_over(directory: Path) -> None:
    """Give ``directory`` and everything in it to the shell's uid; a no-op off root."""
    if os.geteuid() != 0:
        return
    os.chown(directory, SHELL_UID, SHELL_UID)
    for parent, dirs, names in os.walk(directory):
        for name in (*dirs, *names):
            os.chown(Path(parent, name), SHELL_UID, SHELL_UID, follow_symlinks=False)


def reset(root: Path) -> None:
    """Empty ``root`` and create its ``files`` and ``outputs`` directories."""
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for name in (FILES, OUTPUTS):
        (root / name).mkdir()
        _hand_over(root / name)


async def stage(files: DataFiles, refs: Sequence[DataFileRef], root: Path) -> list[dict[str, str]]:
    """Download attachments into ``root/files``; returns their ``data_files`` declarations."""
    paths = await asyncio.gather(*(files.download(ref, root / FILES) for ref in refs))
    _hand_over(root / FILES)
    return [
        {"path": path.relative_to(root).as_posix(), "file_id": ref.file_id}
        for ref, path in zip(refs, paths, strict=True)
    ]


async def stage_bundled(refs: Sequence[str], root: Path) -> list[Path]:
    """Copy bundled attachments into the agent's files directory."""
    paths = [await asyncio.to_thread(copy_bundled, relative, root / FILES) for relative in refs]
    _hand_over(root / FILES)
    return paths


def deliverable_targets(root: Path, names: Sequence[str]) -> list[Path]:
    """Where each named deliverable must be saved, under ``root/outputs``."""
    return [destination(root / OUTPUTS, name) for name in names]


@dataclass(frozen=True)
class Deliverable:
    """A file the task asked for, as the agent left it."""

    path: str
    status: Literal["saved", "missing", "unreadable"]
    text: str
    content: bytes = b""

    def to_json(self) -> dict[str, str]:
        return {"path": self.path, "status": self.status}


def _deliverable_bytes(root: Path, relative: Path) -> bytes:
    if len(relative.parts) < 2 or relative.parts[0] != OUTPUTS or ".." in relative.parts:
        raise UnreadableDocument("is not inside the outputs directory")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    # Keep each directory open: replacing an ancestor must not redirect a privileged read.
    with ExitStack() as opened:
        directory = os.open(root, directory_flags)
        opened.callback(os.close, directory)
        for component in relative.parts[:-1]:
            directory = os.open(component, directory_flags, dir_fd=directory)
            opened.callback(os.close, directory)
        descriptor = os.open(
            relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise UnreadableDocument("is not a regular file")
            if info.st_size > MAX_FILE_BYTES:
                raise UnreadableDocument(f"exceeds the {MAX_FILE_BYTES}-byte input limit")
            content = source.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise UnreadableDocument(f"exceeds the {MAX_FILE_BYTES}-byte input limit")
            return content


def read_deliverable(root: Path, target: Path) -> Deliverable:
    relative = target.relative_to(root)
    path = relative.as_posix()
    try:
        content = _deliverable_bytes(root, relative)
        # Parse a private copy after closing the agent's file, never reopen its pathname.
        with TemporaryDirectory(prefix="hud-deliverable-") as directory:
            snapshot = Path(directory) / target.name
            snapshot.write_bytes(content)
            text = document_text(snapshot)
    except FileNotFoundError:
        return Deliverable(path, "missing", "The agent did not save this file.")
    except UnreadableDocument as error:
        return Deliverable(path, "unreadable", f"The file {error}.")
    except OSError:
        return Deliverable(path, "unreadable", "The file cannot be read without following links.")
    return Deliverable(path, "saved", text, content)


def render(deliverables: Sequence[Deliverable]) -> str:
    return "\n\n".join(f"### {item.path} ({item.status})\n{item.text}" for item in deliverables)
