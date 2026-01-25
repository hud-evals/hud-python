"""Tests for read tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from hud.tools.filesystem import GeminiReadTool, ReadTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with test files."""
    # Create a simple test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n")

    # Create a longer file for pagination tests
    long_file = tmp_path / "long.txt"
    long_file.write_text("\n".join(f"line {i}" for i in range(1, 101)))

    return tmp_path


class TestReadTool:
    """Tests for OpenCode-style ReadTool."""

    @pytest.mark.asyncio
    async def test_read_simple_file(self, workspace: Path) -> None:
        """Test reading a simple file."""
        tool = ReadTool(base_path=str(workspace))
        result = await tool(filePath=str(workspace / "test.txt"))

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "<file>" in result[0].text
        assert "</file>" in result[0].text
        assert "00001|" in result[0].text
        assert "line 1" in result[0].text

    @pytest.mark.asyncio
    async def test_read_with_offset(self, workspace: Path) -> None:
        """Test reading with offset."""
        tool = ReadTool(base_path=str(workspace))
        result = await tool(filePath=str(workspace / "test.txt"), offset=2)

        text = result[0].text
        assert "line 3" in text
        assert "00003|" in text

    @pytest.mark.asyncio
    async def test_read_with_limit(self, workspace: Path) -> None:
        """Test reading with limit."""
        tool = ReadTool(base_path=str(workspace))
        result = await tool(filePath=str(workspace / "long.txt"), limit=5)

        text = result[0].text
        assert "line 1" in text
        assert "line 5" in text
        # Should indicate more content available
        assert "File has more lines" in text or "more lines" in text.lower()

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, workspace: Path) -> None:
        """Test reading non-existent file raises error."""
        from hud.tools.types import ToolError

        tool = ReadTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="not found"):
            await tool(filePath=str(workspace / "nonexistent.txt"))

    @pytest.mark.asyncio
    async def test_read_directory_error(self, workspace: Path) -> None:
        """Test reading a directory raises error."""
        from hud.tools.types import ToolError

        tool = ReadTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="directory"):
            await tool(filePath=str(workspace))


class TestGeminiReadTool:
    """Tests for Gemini CLI-style GeminiReadTool."""

    @pytest.mark.asyncio
    async def test_read_simple_file(self, workspace: Path) -> None:
        """Test reading a simple file."""
        tool = GeminiReadTool(base_path=str(workspace))
        result = await tool(file_path=str(workspace / "test.txt"))

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "line 1" in result[0].text

    @pytest.mark.asyncio
    async def test_read_with_pagination(self, workspace: Path) -> None:
        """Test reading with offset and limit."""
        tool = GeminiReadTool(base_path=str(workspace))
        result = await tool(
            file_path=str(workspace / "long.txt"),
            offset=10,
            limit=5,
        )

        text = result[0].text
        assert "IMPORTANT" in text  # Truncation warning
        assert "line 11" in text

    @pytest.mark.asyncio
    async def test_read_empty_path_error(self, workspace: Path) -> None:
        """Test empty path raises error."""
        from hud.tools.types import ToolError

        tool = GeminiReadTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="non-empty"):
            await tool(file_path="")

    @pytest.mark.asyncio
    async def test_read_negative_offset_error(self, workspace: Path) -> None:
        """Test negative offset raises error."""
        from hud.tools.types import ToolError

        tool = GeminiReadTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="non-negative"):
            await tool(file_path=str(workspace / "test.txt"), offset=-1)
