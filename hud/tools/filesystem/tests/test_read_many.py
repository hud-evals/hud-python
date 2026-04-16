"""Tests for GeminiReadManyTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from hud.tools.filesystem import GeminiReadManyTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with test files."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text("def main():\n    pass\n")
    (src / "utils.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "readme.txt").write_text("Hello world\n")
    (tmp_path / "config.json").write_text('{"key": "value"}\n')

    # Create a node_modules dir to test default excludes
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "pkg.js").write_text("// package\n")

    return tmp_path


class TestGeminiReadManyTool:
    """Tests for GeminiReadManyTool."""

    def test_init(self) -> None:
        """Test initialization."""
        tool = GeminiReadManyTool(base_path=".")
        assert tool.name == "read_many_files"

    @pytest.mark.asyncio
    async def test_read_single_file(self, workspace: Path) -> None:
        """Test reading a single literal file path."""
        tool = GeminiReadManyTool(base_path=str(workspace))
        result = await tool(include=["readme.txt"])

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        text = result[0].text
        assert "readme.txt" in text
        assert "Hello world" in text

    @pytest.mark.asyncio
    async def test_read_glob_pattern(self, workspace: Path) -> None:
        """Test reading files by glob pattern."""
        tool = GeminiReadManyTool(base_path=str(workspace))
        result = await tool(include=["**/*.py"])

        text = result[0].text
        assert "main.py" in text
        assert "utils.py" in text
        assert "def main" in text
        assert "def helper" in text

    @pytest.mark.asyncio
    async def test_read_with_exclude(self, workspace: Path) -> None:
        """Test excluding files by pattern."""
        tool = GeminiReadManyTool(base_path=str(workspace))
        result = await tool(include=["**/*.py"], exclude=["**/utils.py"])

        text = result[0].text
        assert "main.py" in text
        assert "utils.py" not in text

    @pytest.mark.asyncio
    async def test_default_excludes(self, workspace: Path) -> None:
        """Test that node_modules is excluded by default."""
        tool = GeminiReadManyTool(base_path=str(workspace))
        result = await tool(include=["**/*.js"])

        text = result[0].text
        assert "pkg.js" not in text

    @pytest.mark.asyncio
    async def test_no_default_excludes(self, workspace: Path) -> None:
        """Test disabling default excludes."""
        tool = GeminiReadManyTool(base_path=str(workspace))
        result = await tool(include=["**/*.js"], useDefaultExcludes=False)

        text = result[0].text
        assert "pkg.js" in text

    @pytest.mark.asyncio
    async def test_no_matches(self, workspace: Path) -> None:
        """Test when no files match."""
        tool = GeminiReadManyTool(base_path=str(workspace))
        result = await tool(include=["**/*.xyz"])

        text = result[0].text
        assert "No files found" in text

    @pytest.mark.asyncio
    async def test_file_separators(self, workspace: Path) -> None:
        """Test output has file path separators."""
        tool = GeminiReadManyTool(base_path=str(workspace))
        result = await tool(include=["**/*.py"])

        text = result[0].text
        assert "---" in text
        assert "End of content" in text

    @pytest.mark.asyncio
    async def test_empty_include_error(self, workspace: Path) -> None:
        """Test empty include raises error."""
        from hud.tools.types import ToolError

        tool = GeminiReadManyTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="non-empty"):
            await tool(include=[])
