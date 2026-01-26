"""Tests for list tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hud.tools.filesystem import GeminiListTool, ListTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with directory structure."""
    # Create directories
    src = tmp_path / "src"
    src.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()

    # Create files
    (tmp_path / "README.md").write_text("# README")
    (src / "main.py").write_text("# main")
    (src / "utils.py").write_text("# utils")
    (docs / "guide.md").write_text("# Guide")

    return tmp_path


class TestListTool:
    """Tests for OpenCode-style ListTool."""

    @pytest.mark.asyncio
    async def test_list_directory(self, workspace: Path) -> None:
        """Test listing directory contents."""
        tool = ListTool(base_path=str(workspace))
        result = await tool(path=str(workspace))

        text = result[0].text
        assert "src" in text
        assert "docs" in text
        assert "README.md" in text

    @pytest.mark.asyncio
    async def test_list_subdirectory(self, workspace: Path) -> None:
        """Test listing subdirectory."""
        tool = ListTool(base_path=str(workspace))
        result = await tool(path=str(workspace / "src"))

        text = result[0].text
        assert "main.py" in text
        assert "utils.py" in text

    @pytest.mark.asyncio
    async def test_list_with_ignore(self, workspace: Path) -> None:
        """Test listing with ignore patterns."""
        tool = ListTool(base_path=str(workspace))
        result = await tool(path=str(workspace), ignore=["*.md"])

        text = result[0].text
        assert "README.md" not in text
        assert "src" in text

    @pytest.mark.asyncio
    async def test_list_nonexistent_path(self, workspace: Path) -> None:
        """Test listing non-existent path."""
        from hud.tools.types import ToolError

        tool = ListTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="not found"):
            await tool(path=str(workspace / "nonexistent"))

    @pytest.mark.asyncio
    async def test_list_tree_format(self, workspace: Path) -> None:
        """Test that output is in tree format with indentation."""
        tool = ListTool(base_path=str(workspace))
        result = await tool(path=str(workspace))

        text = result[0].text
        # Should have indented entries
        assert "  " in text


class TestGeminiListTool:
    """Tests for Gemini CLI-style GeminiListTool."""

    @pytest.mark.asyncio
    async def test_list_directory(self, workspace: Path) -> None:
        """Test listing directory contents."""
        tool = GeminiListTool(base_path=str(workspace))
        result = await tool(dir_path=str(workspace))

        text = result[0].text
        assert "DIR" in text  # Directories marked with DIR

    @pytest.mark.asyncio
    async def test_list_dir_prefix(self, workspace: Path) -> None:
        """Test that directories have DIR prefix."""
        tool = GeminiListTool(base_path=str(workspace))
        result = await tool(dir_path=str(workspace))

        text = result[0].text
        assert "DIR  src" in text or "DIR  docs" in text

    @pytest.mark.asyncio
    async def test_list_empty_path_error(self, workspace: Path) -> None:
        """Test empty path raises error."""
        from hud.tools.types import ToolError

        tool = GeminiListTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="non-empty"):
            await tool(dir_path="")
