"""Tests for the legacy apply_patch compatibility wrapper."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from mcp.types import TextContent

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
