"""Tests for edit tool."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import TextContent

from hud.tools.coding import EditTool
from hud.tools.types import ToolError


class TestEditTool:
    """Tests for EditTool."""

    def test_edit_tool_init(self):
        """Test EditTool initialization."""
        tool = EditTool()
        assert tool is not None

    @pytest.mark.asyncio
    async def test_validate_path_not_absolute(self):
        """Test validate_path with non-absolute path."""
        tool = EditTool()

        with pytest.raises(ToolError) as exc_info:
            tool.validate_path("create", Path("relative/path.txt"))

        assert "not an absolute path" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_path_not_exists(self):
        """Test validate_path when file doesn't exist for non-create commands."""
        tool = EditTool()

        # Use a platform-appropriate absolute path
        if sys.platform == "win32":
            nonexistent_path = Path("C:\\nonexistent\\file.txt")
        else:
            nonexistent_path = Path("/nonexistent/file.txt")

        with pytest.raises(ToolError) as exc_info:
            tool.validate_path("view", nonexistent_path)

        assert "does not exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_path_exists_for_create(self):
        """Test validate_path when file exists for create command."""
        tool = EditTool()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(ToolError) as exc_info:
                tool.validate_path("create", tmp_path)

            assert "already exists" in str(exc_info.value)
        finally:
            os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_create_file(self):
        """Test creating a new file."""
        tool = EditTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            content = "Hello, World!"

            # Patch the module-level write_file_async function
            with patch(
                "hud.tools.coding.edit.write_file_async", new_callable=AsyncMock
            ) as mock_write:
                result = await tool(command="create", path=str(file_path), file_text=content)

                assert isinstance(result, list)
                assert len(result) > 0
                # For TextContent, we need to check the text attribute
                text_blocks = [block for block in result if isinstance(block, TextContent)]
                assert len(text_blocks) > 0
                assert "created successfully" in text_blocks[0].text
                mock_write.assert_called_once_with(file_path, content)

    @pytest.mark.asyncio
    async def test_create_file_no_text(self):
        """Test creating file without file_text raises error."""
        tool = EditTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"

            with pytest.raises(ToolError) as exc_info:
                await tool(command="create", path=str(file_path))

            assert "file_text` is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_view_file(self):
        """Test viewing a file."""
        tool = EditTool()

        file_content = "Line 1\nLine 2\nLine 3"

        # Patch module-level functions
        with (
            patch("hud.tools.coding.edit.read_file_async", new_callable=AsyncMock) as mock_read,
            patch.object(tool, "validate_path"),
        ):
            mock_read.return_value = file_content

            result = await tool(command="view", path="/tmp/test.txt")

            assert isinstance(result, list)
            assert len(result) > 0
            text_blocks = [block for block in result if isinstance(block, TextContent)]
            assert len(text_blocks) > 0
            combined_text = "".join(block.text for block in text_blocks)
            assert "Line 1" in combined_text
            assert "Line 2" in combined_text
            assert "Line 3" in combined_text

    @pytest.mark.asyncio
    async def test_view_with_range(self):
        """Test viewing a file with line range."""
        tool = EditTool()

        file_content = "\n".join([f"Line {i}" for i in range(1, 11)])

        with (
            patch("hud.tools.coding.edit.read_file_async", new_callable=AsyncMock) as mock_read,
            patch.object(tool, "validate_path"),
        ):
            mock_read.return_value = file_content

            result = await tool(command="view", path="/tmp/test.txt", view_range=[3, 5])

            assert isinstance(result, list)
            assert len(result) > 0
            text_blocks = [block for block in result if isinstance(block, TextContent)]
            assert len(text_blocks) > 0
            combined_text = "".join(block.text for block in text_blocks)
            # Lines 3-5 should be in output (using tab format)
            assert "3\tLine 3" in combined_text
            assert "4\tLine 4" in combined_text
            assert "5\tLine 5" in combined_text
            # Line 1 and 10 should not be in output (outside range)
            assert "1\tLine 1" not in combined_text
            assert "10\tLine 10" not in combined_text

    @pytest.mark.asyncio
    async def test_str_replace_success(self):
        """Test successful string replacement."""
        tool = EditTool()

        file_content = "Hello, World!\nThis is a test."
        expected_content = "Hello, Universe!\nThis is a test."

        with (
            patch("hud.tools.coding.edit.read_file_async", new_callable=AsyncMock) as mock_read,
            patch("hud.tools.coding.edit.write_file_async", new_callable=AsyncMock) as mock_write,
            patch.object(tool, "validate_path"),
        ):
            mock_read.return_value = file_content

            result = await tool(
                command="str_replace", path="/tmp/test.txt", old_str="World", new_str="Universe"
            )

            assert isinstance(result, list)
            assert len(result) > 0
            text_blocks = [block for block in result if isinstance(block, TextContent)]
            assert len(text_blocks) > 0
            combined_text = "".join(block.text for block in text_blocks)
            assert "has been edited" in combined_text
            mock_write.assert_called_once_with(Path("/tmp/test.txt"), expected_content)

    @pytest.mark.asyncio
    async def test_str_replace_not_found(self):
        """Test string replacement when old_str not found."""
        tool = EditTool()

        file_content = "Hello, World!"

        with (
            patch("hud.tools.coding.edit.read_file_async", new_callable=AsyncMock) as mock_read,
            patch.object(tool, "validate_path"),
        ):
            mock_read.return_value = file_content

            with pytest.raises(ToolError) as exc_info:
                await tool(
                    command="str_replace",
                    path="/tmp/test.txt",
                    old_str="Universe",
                    new_str="Galaxy",
                )

            assert "did not appear verbatim" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_str_replace_multiple_occurrences(self):
        """Test string replacement with multiple occurrences."""
        tool = EditTool()

        file_content = "Test test\nAnother test line"

        with (
            patch("hud.tools.coding.edit.read_file_async", new_callable=AsyncMock) as mock_read,
            patch.object(tool, "validate_path"),
        ):
            mock_read.return_value = file_content

            with pytest.raises(ToolError) as exc_info:
                await tool(
                    command="str_replace", path="/tmp/test.txt", old_str="test", new_str="example"
                )

            assert "Multiple occurrences" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_command(self):
        """Test invalid command raises error."""
        tool = EditTool()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            # Create the file so validate_path doesn't fail
            file_path.write_text("test content")

            with pytest.raises((ToolError, AttributeError)) as exc_info:
                await tool(
                    command="invalid_command",  # type: ignore
                    path=str(file_path),
                )

            error_msg = str(exc_info.value)
            assert "Unrecognized command" in error_msg or "name" in error_msg
