"""Tests for grep/search tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from hud.tools.filesystem import GeminiSearchTool, GrepTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with test files."""
    # Python file
    py_file = tmp_path / "example.py"
    py_file.write_text("def hello():\n    print('Hello World')\n\ndef goodbye():\n    pass\n")

    # JavaScript file
    js_file = tmp_path / "app.js"
    js_file.write_text("function main() {\n  console.log('Hello');\n}\n")

    # Text file
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("TODO: fix this\nFIXME: urgent\nTODO: later\n")

    return tmp_path


class TestGrepTool:
    """Tests for OpenCode-style GrepTool."""

    @pytest.mark.asyncio
    async def test_grep_simple_pattern(self, workspace: Path) -> None:
        """Test searching for a simple pattern."""
        tool = GrepTool(base_path=str(workspace))
        result = await tool(pattern="def")

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "def hello" in result[0].text or "def goodbye" in result[0].text

    @pytest.mark.asyncio
    async def test_grep_with_include_filter(self, workspace: Path) -> None:
        """Test searching with file type filter."""
        tool = GrepTool(base_path=str(workspace))
        result = await tool(pattern="Hello", include="*.py")

        text = result[0].text
        assert "example.py" in text
        assert "app.js" not in text

    @pytest.mark.asyncio
    async def test_grep_regex_pattern(self, workspace: Path) -> None:
        """Test searching with regex pattern."""
        tool = GrepTool(base_path=str(workspace))
        result = await tool(pattern="TODO|FIXME")

        text = result[0].text
        assert "TODO" in text
        assert "FIXME" in text

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, workspace: Path) -> None:
        """Test search with no matches."""
        tool = GrepTool(base_path=str(workspace))
        result = await tool(pattern="nonexistent_pattern_xyz")

        assert "No files found" in result[0].text

    @pytest.mark.asyncio
    async def test_grep_invalid_regex(self, workspace: Path) -> None:
        """Test invalid regex raises error."""
        from hud.tools.types import ToolError

        tool = GrepTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="Invalid regex"):
            await tool(pattern="[invalid")


class TestGeminiSearchTool:
    """Tests for Gemini CLI-style GeminiSearchTool."""

    @pytest.mark.asyncio
    async def test_search_simple_pattern(self, workspace: Path) -> None:
        """Test searching for a simple pattern."""
        tool = GeminiSearchTool(base_path=str(workspace))
        result = await tool(pattern="def")

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Found" in result[0].text

    @pytest.mark.asyncio
    async def test_search_groups_by_file(self, workspace: Path) -> None:
        """Test that results are grouped by file."""
        tool = GeminiSearchTool(base_path=str(workspace))
        result = await tool(pattern="TODO")

        text = result[0].text
        assert "notes.txt:" in text
        assert "Line" in text

    @pytest.mark.asyncio
    async def test_search_no_matches(self, workspace: Path) -> None:
        """Test search with no matches."""
        tool = GeminiSearchTool(base_path=str(workspace))
        result = await tool(pattern="nonexistent_pattern_xyz")

        assert "No matches found" in result[0].text
