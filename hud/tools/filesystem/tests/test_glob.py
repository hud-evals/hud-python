"""Tests for glob tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from hud.tools.filesystem import GeminiGlobTool, GlobTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with test files."""
    # Create directory structure
    src = tmp_path / "src"
    src.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()

    # Python files
    (src / "main.py").write_text("# main")
    (src / "utils.py").write_text("# utils")
    (tests / "test_main.py").write_text("# test")

    # JavaScript files
    (src / "app.js").write_text("// app")
    (src / "config.json").write_text("{}")

    return tmp_path


class TestGlobTool:
    """Tests for OpenCode-style GlobTool."""

    @pytest.mark.asyncio
    async def test_glob_python_files(self, workspace: Path) -> None:
        """Test finding Python files."""
        tool = GlobTool(base_path=str(workspace))
        result = await tool(pattern="**/*.py")

        text = result[0].text
        assert "main.py" in text
        assert "utils.py" in text
        assert "test_main.py" in text

    @pytest.mark.asyncio
    async def test_glob_in_subdirectory(self, workspace: Path) -> None:
        """Test globbing in a subdirectory."""
        tool = GlobTool(base_path=str(workspace))
        result = await tool(pattern="*.py", path="src")

        text = result[0].text
        assert "main.py" in text
        assert "test_main.py" not in text

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, workspace: Path) -> None:
        """Test glob with no matches."""
        tool = GlobTool(base_path=str(workspace))
        result = await tool(pattern="**/*.xyz")

        assert "No files found" in result[0].text

    @pytest.mark.asyncio
    async def test_glob_nonexistent_path(self, workspace: Path) -> None:
        """Test glob with non-existent path."""
        from hud.tools.types import ToolError

        tool = GlobTool(base_path=str(workspace))
        with pytest.raises(ToolError, match="not found"):
            await tool(pattern="*.py", path="nonexistent")


class TestGeminiGlobTool:
    """Tests for Gemini CLI-style GeminiGlobTool."""

    @pytest.mark.asyncio
    async def test_glob_returns_absolute_paths(self, workspace: Path) -> None:
        """Test that results are absolute paths."""
        tool = GeminiGlobTool(base_path=str(workspace))
        result = await tool(pattern="**/*.py")

        text = result[0].text
        # Gemini style returns absolute paths
        lines = text.strip().split("\n")
        for line in lines:
            if line and not line.startswith("("):
                assert Path(line).is_absolute() or str(workspace) in line

    @pytest.mark.asyncio
    async def test_glob_alphabetical_sort(self, workspace: Path) -> None:
        """Test that results are sorted alphabetically."""
        tool = GeminiGlobTool(base_path=str(workspace))
        result = await tool(pattern="**/*.py")

        text = result[0].text
        lines = [line for line in text.strip().split("\n") if line and not line.startswith("(")]
        # Should be sorted alphabetically
        assert lines == sorted(lines)
