"""Tests for Gemini-style coding tools."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from hud.tools.coding.gemini_edit import GeminiEditTool
from hud.tools.coding.gemini_shell import GeminiShellTool
from hud.tools.types import ToolError


class TestGeminiShellTool:
    """Tests for GeminiShellTool."""

    def test_init(self) -> None:
        """Test initialization."""
        tool = GeminiShellTool()
        assert tool.name == "run_shell_command"
        assert tool._base_directory == os.path.abspath(".")

    def test_init_with_base_directory(self) -> None:
        """Test initialization with custom base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = GeminiShellTool(base_directory=tmpdir)
            assert tool._base_directory == os.path.abspath(tmpdir)

    def test_resolve_directory_none(self) -> None:
        """Test directory resolution with None."""
        tool = GeminiShellTool()
        assert tool._resolve_directory(None) == tool._base_directory

    def test_resolve_directory_relative(self) -> None:
        """Test directory resolution with relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = GeminiShellTool(base_directory=tmpdir)
            result = tool._resolve_directory("subdir")
            assert result == os.path.normpath(os.path.join(tmpdir, "subdir"))

    def test_resolve_directory_absolute(self) -> None:
        """Test directory resolution with absolute path."""
        tool = GeminiShellTool()
        abs_path = "/tmp/test"
        result = tool._resolve_directory(abs_path)
        assert result == abs_path

    @pytest.mark.asyncio
    async def test_call_no_command(self) -> None:
        """Test call with no command raises error."""
        tool = GeminiShellTool()
        with pytest.raises(ToolError, match="Command cannot be empty"):
            await tool(command="")

    @pytest.mark.asyncio
    async def test_call_simple_command(self) -> None:
        """Test simple command execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = GeminiShellTool(base_directory=tmpdir)
            result = await tool(command="echo hello")
            # Returns list of ContentBlock
            assert len(result) >= 1
            assert hasattr(result[0], "text")
            assert "hello" in result[0].text  # type: ignore[union-attr]


class TestGeminiEditTool:
    """Tests for GeminiEditTool."""

    def test_init(self) -> None:
        """Test initialization."""
        tool = GeminiEditTool()
        assert tool.name == "replace"

    def test_init_with_base_directory(self) -> None:
        """Test initialization with custom base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = GeminiEditTool(base_directory=tmpdir)
            assert tool._base_directory == str(Path(tmpdir).resolve())

    def test_resolve_path_relative(self) -> None:
        """Test path resolution with relative path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = GeminiEditTool(base_directory=tmpdir)
            result = tool._resolve_path("test.txt")
            assert result == Path(tmpdir) / "test.txt"

    def test_resolve_path_absolute(self) -> None:
        """Test path resolution with absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = GeminiEditTool()
            # Use a platform-appropriate absolute path
            abs_path = str(Path(tmpdir) / "test.txt")
            result = tool._resolve_path(abs_path)
            assert result == Path(abs_path)
            assert result.is_absolute()

    @pytest.mark.asyncio
    async def test_call_missing_file_path(self) -> None:
        """Test call with missing file_path raises error."""
        tool = GeminiEditTool()
        with pytest.raises(ToolError, match=r"file_path.*must be non-empty"):
            await tool(
                file_path="",
                instruction="test",
                old_string="old",
                new_string="new",
            )

    @pytest.mark.asyncio
    async def test_call_missing_instruction(self) -> None:
        """Test call with missing instruction raises error."""
        tool = GeminiEditTool()
        with pytest.raises(ToolError, match=r"instruction.*must be non-empty"):
            await tool(
                file_path="test.txt",
                instruction="",
                old_string="old",
                new_string="new",
            )

    @pytest.mark.asyncio
    async def test_call_file_not_found(self) -> None:
        """Test call with nonexistent file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = GeminiEditTool(base_directory=tmpdir)
            with pytest.raises(ToolError, match="File not found"):
                await tool(
                    file_path="nonexistent.txt",
                    instruction="test edit",
                    old_string="old",
                    new_string="new",
                )

    @pytest.mark.asyncio
    async def test_call_old_string_not_found(self) -> None:
        """Test call with old_string not in file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello world")

            tool = GeminiEditTool(base_directory=tmpdir)
            with pytest.raises(ToolError, match="0 occurrences found"):
                await tool(
                    file_path="test.txt",
                    instruction="test edit",
                    old_string="foo bar",
                    new_string="new",
                )

    @pytest.mark.asyncio
    async def test_call_multiple_occurrences_no_expected(self) -> None:
        """Test call with multiple occurrences without expected_replacements replaces first only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello hello hello")

            tool = GeminiEditTool(base_directory=tmpdir)
            # Without expected_replacements, it replaces only the first occurrence
            result = await tool(
                file_path="test.txt",
                instruction="test edit",
                old_string="hello",
                new_string="world",
            )
            assert test_file.read_text() == "world hello hello"
            assert "1 replacements" in result[0].text  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_call_successful_edit(self) -> None:
        """Test successful file edit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello world")

            tool = GeminiEditTool(base_directory=tmpdir)
            result = await tool(
                file_path="test.txt",
                instruction="Replace hello with goodbye",
                old_string="hello",
                new_string="goodbye",
            )

            # Verify file was modified
            assert test_file.read_text() == "goodbye world"

            # Verify result message
            assert len(result) == 1
            assert hasattr(result[0], "text")
            assert "Successfully modified" in result[0].text  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_call_multiple_replacements(self) -> None:
        """Test multiple replacements with expected_replacements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello hello hello")

            tool = GeminiEditTool(base_directory=tmpdir)
            await tool(
                file_path="test.txt",
                instruction="Replace all hello with world",
                old_string="hello",
                new_string="world",
                expected_replacements=3,
            )

            # Verify file was modified
            assert test_file.read_text() == "world world world"

    @pytest.mark.asyncio
    async def test_file_history_saved(self) -> None:
        """Test that file history is saved for potential undo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("original content")

            tool = GeminiEditTool(base_directory=tmpdir)
            await tool(
                file_path="test.txt",
                instruction="test edit",
                old_string="original",
                new_string="modified",
            )

            # Check history was saved
            assert len(tool._file_history[test_file]) == 1
            assert tool._file_history[test_file][0] == "original content"
