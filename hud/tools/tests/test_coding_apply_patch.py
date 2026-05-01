"""Tests for apply_patch compatibility tool and patch parser helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from mcp.types import TextContent

from hud.agents.openai.tools.apply_patch import (
    ActionType,
    DiffError,
    Parser,
    _apply_commit,
    _identify_files_needed,
    _patch_to_commit,
    _text_to_patch,
)
from hud.tools._legacy import ApplyPatchTool
from hud.tools.coding import EditTool


class TestApplyPatchTool:
    """Tests for ApplyPatchTool compatibility wrapper."""

    def test_apply_patch_tool_is_edit_tool(self):
        tool = ApplyPatchTool()
        assert isinstance(tool, EditTool)
        assert tool.name == "edit"
        assert "native_tools" not in tool.meta

    @pytest.mark.asyncio
    async def test_update_file_uses_edit_tool_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("old\n")

            result = await tool(command="write", path="test.txt", file_text="new\n")

            assert file_path.read_text() == "new\n"
            assert isinstance(result[0], TextContent)
            assert "written successfully" in result[0].text


class TestPatchParser:
    """Focused tests for shared V4A parser helpers used by EditTool."""

    def test_parse_add_file(self):
        lines = [
            "*** Begin Patch",
            "*** Add File: new.txt",
            "+line 1",
            "+line 2",
            "*** End Patch",
        ]
        parser = Parser(current_files={}, lines=lines, index=1)
        parser.parse()

        action = parser.patch.actions["new.txt"]
        assert action.type == ActionType.ADD
        assert action.new_file == "line 1\nline 2"

    def test_parse_update_file(self):
        text = "*** Begin Patch\n*** Update File: test.txt\n@@\n-old\n+new\n*** End Patch"

        patch, fuzz = _text_to_patch(text, {"test.txt": "old\n"})

        assert fuzz == 0
        action = patch.actions["test.txt"]
        assert action.type == ActionType.UPDATE

    def test_identify_files_needed(self):
        text = "*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new\n*** End Patch"
        assert _identify_files_needed(text) == ["a.txt"]

    def test_apply_commit_update(self):
        patch, _ = _text_to_patch(
            "*** Begin Patch\n*** Update File: a.txt\n@@\n-old\n+new\n*** End Patch",
            {"a.txt": "old\n"},
        )
        commit = _patch_to_commit(patch, {"a.txt": "old\n"})
        files = {"a.txt": "old\n"}

        def write(path: str, content: str | None) -> None:
            files[path] = content or ""

        def remove(path: str) -> None:
            del files[path]

        _apply_commit(commit, write, remove)
        assert files["a.txt"] == "new\n"

    def test_invalid_patch_raises(self):
        with pytest.raises(DiffError):
            _text_to_patch("not a patch", {})
