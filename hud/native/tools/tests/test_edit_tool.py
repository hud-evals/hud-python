"""``EditTool`` — local file view/create/replace/insert/delete/undo over a base path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.types import ToolError
from hud.native.tools.coding.edit import EditTool

if TYPE_CHECKING:
    from pathlib import Path


def _text(blocks: list[Any]) -> str:
    return "\n".join(getattr(b, "text", "") for b in blocks)


async def test_create_then_read(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    await tool(command="create", path="f.txt", file_text="hello world")

    assert (tmp_path / "f.txt").read_text() == "hello world"
    assert "hello world" in _text(await tool(command="read", path="f.txt"))


async def test_replace_unique_fragment(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    (tmp_path / "f.txt").write_text("alpha beta gamma")

    await tool(command="replace", path="f.txt", old_text="beta", new_text="BETA")

    assert (tmp_path / "f.txt").read_text() == "alpha BETA gamma"


async def test_replace_ambiguous_fragment_errors(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    (tmp_path / "f.txt").write_text("x x x")

    with pytest.raises(ToolError, match="Multiple occurrences"):
        await tool(command="replace", path="f.txt", old_text="x", new_text="y")


async def test_insert_after_line(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    (tmp_path / "f.txt").write_text("line1\nline2\n")

    await tool(command="insert", path="f.txt", insert_line=1, insert_text="inserted")

    assert (tmp_path / "f.txt").read_text().splitlines()[1] == "inserted"


async def test_undo_restores_previous_content(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    (tmp_path / "f.txt").write_text("v1")

    await tool(command="replace", path="f.txt", old_text="v1", new_text="v2")
    assert (tmp_path / "f.txt").read_text() == "v2"

    await tool(command="undo", path="f.txt")
    assert (tmp_path / "f.txt").read_text() == "v1"


async def test_delete_removes_file(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    (tmp_path / "f.txt").write_text("bye")

    await tool(command="delete", path="f.txt")

    assert not (tmp_path / "f.txt").exists()


async def test_create_over_existing_errors(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    (tmp_path / "f.txt").write_text("here")

    with pytest.raises(ToolError, match="already exists"):
        await tool(command="create", path="f.txt", file_text="nope")


async def test_missing_command_errors(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    with pytest.raises(ToolError, match="command"):
        await tool(path="f.txt")


async def test_path_traversal_blocked(tmp_path: Path) -> None:
    tool = EditTool(base_path=tmp_path)
    with pytest.raises(ToolError, match="traversal"):
        await tool(command="create", path="../escape.txt", file_text="x")
