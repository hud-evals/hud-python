"""``MemoryTool`` — file-backed persistent memory operations under /memories."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.types import ToolError
from hud.native.tools.memory import MemoryTool

if TYPE_CHECKING:
    from pathlib import Path


def _text(blocks: list[Any]) -> str:
    return " ".join(getattr(b, "text", "") for b in blocks)


async def test_create_and_view_file(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    await mt(command="create", path="/memories/notes.md", file_text="hello\n")

    assert (tmp_path / "mem" / "notes.md").read_text(encoding="utf-8") == "hello\n"
    blocks = await mt(command="view", path="/memories/notes.md")
    assert "hello" in _text(blocks)


async def test_view_directory_lists_files(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    await mt(command="create", path="/memories/a.md", file_text="x")

    blocks = await mt(command="view", path="/memories")
    assert "a.md" in _text(blocks)


async def test_str_replace_rewrites_content(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    await mt(command="create", path="/memories/n.md", file_text="hello world")

    await mt(command="str_replace", path="/memories/n.md", old_str="world", new_str="there")
    assert (tmp_path / "mem" / "n.md").read_text(encoding="utf-8") == "hello there"


async def test_insert_adds_line(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    await mt(command="create", path="/memories/n.md", file_text="line1\n")

    await mt(command="insert", path="/memories/n.md", insert_line=1, insert_text="line2")
    assert "line2" in (tmp_path / "mem" / "n.md").read_text(encoding="utf-8")


async def test_rename_then_delete(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    await mt(command="create", path="/memories/old.md", file_text="x")

    await mt(command="rename", old_path="/memories/old.md", new_path="/memories/new.md")
    assert (tmp_path / "mem" / "new.md").exists()
    assert not (tmp_path / "mem" / "old.md").exists()

    await mt(command="delete", path="/memories/new.md")
    assert not (tmp_path / "mem" / "new.md").exists()


async def test_create_requires_file_text(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    with pytest.raises(ToolError):
        await mt(command="create", path="/memories/x.md")


async def test_str_replace_missing_file_errors(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    with pytest.raises(ToolError):
        await mt(command="str_replace", path="/memories/missing.md", old_str="a", new_str="b")


async def test_create_over_existing_errors(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    await mt(command="create", path="/memories/dup.md", file_text="a")
    with pytest.raises(ToolError):
        await mt(command="create", path="/memories/dup.md", file_text="b")


async def test_unrecognized_command_errors(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    with pytest.raises(ToolError):
        await mt(command="bogus")  # type: ignore[arg-type]


async def test_path_traversal_blocked(tmp_path: Path) -> None:
    mt = MemoryTool(memories_dir=tmp_path / "mem")
    with pytest.raises(ValueError, match="traversal"):
        await mt(command="create", path="/memories/../escape.md", file_text="x")
