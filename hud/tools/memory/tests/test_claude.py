"""Tests for Claude Memory Tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import pytest
from mcp.types import TextContent

from hud.tools.memory.claude import ClaudeMemoryCommand, ClaudeMemoryTool
from hud.tools.native_types import NativeToolSpec
from hud.tools.types import ToolError
from hud.types import AgentType

if TYPE_CHECKING:
    from pathlib import Path


class TestClaudeMemoryToolInit:
    """Tests for ClaudeMemoryTool initialization."""

    def test_default_init(self, tmp_path: Path) -> None:
        """Test default initialization."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        assert tool.name == "memory"
        assert tool.title == "Memory"
        assert "persistent" in tool.description.lower()

    def test_custom_memories_dir(self, tmp_path: Path) -> None:
        """Test initialization with custom memories directory."""
        memories_dir = tmp_path / "custom_memories"
        memories_dir.mkdir()
        tool = ClaudeMemoryTool(memories_dir=str(memories_dir))
        assert tool._base_path == memories_dir

    def test_native_specs(self, tmp_path: Path) -> None:
        """Test native spec configuration."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        assert AgentType.CLAUDE in tool.native_specs
        spec = tool.native_specs[AgentType.CLAUDE]
        assert isinstance(spec, NativeToolSpec)
        assert spec.api_type == "memory_20250818"
        assert spec.api_name == "memory"
        assert spec.role == "memory"
        assert spec.beta == "context-management-2025-06-27"


class TestClaudeMemoryView:
    """Test view command."""

    @pytest.mark.asyncio
    async def test_view_empty_directory(self, tmp_path: Path) -> None:
        """Test viewing an empty directory."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        result = await tool(command="view", path="/memories")

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "files and directories" in result[0].text

    @pytest.mark.asyncio
    async def test_view_directory_with_files(self, tmp_path: Path) -> None:
        """Test viewing a directory with files."""
        (tmp_path / "notes.txt").write_text("Some notes here")
        (tmp_path / "data.json").write_text('{"key": "value"}')

        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        result = await tool(command="view", path="/memories")

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        text = result[0].text
        assert "notes.txt" in text
        assert "data.json" in text

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="read_file_async uses shell commands not available on Windows",
    )
    async def test_view_file_content(self, tmp_path: Path) -> None:
        """Test viewing a file's content."""
        content = "Line 1\nLine 2\nLine 3"
        (tmp_path / "test.txt").write_text(content)

        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        result = await tool(command="view", path="/memories/test.txt")

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Line 1" in result[0].text

    @pytest.mark.asyncio
    async def test_view_nonexistent_path(self, tmp_path: Path) -> None:
        """Test viewing a nonexistent path."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="does not exist"):
            await tool(command="view", path="/memories/nonexistent.txt")

    @pytest.mark.asyncio
    async def test_view_default_path(self, tmp_path: Path) -> None:
        """Test view with no path defaults to /memories."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        result = await tool(command="view")

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "files and directories" in result[0].text


class TestClaudeMemoryCreate:
    """Test create command."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="write_file_async uses shell commands not available on Windows",
    )
    async def test_create_file(self, tmp_path: Path) -> None:
        """Test creating a new file."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        result = await tool(
            command="create",
            path="/memories/new_file.txt",
            file_text="Hello, World!",
        )

        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "created successfully" in result[0].text

        # Verify file was created
        created_file = tmp_path / "new_file.txt"
        assert created_file.exists()
        # write_file_async uses heredoc which adds trailing newline
        assert created_file.read_text().rstrip("\n") == "Hello, World!"

    @pytest.mark.asyncio
    async def test_create_existing_file_error(self, tmp_path: Path) -> None:
        """Test creating a file that already exists."""
        existing = tmp_path / "exists.txt"
        existing.write_text("Existing content")

        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="already exists"):
            await tool(command="create", path="/memories/exists.txt", file_text="New content")

    @pytest.mark.asyncio
    async def test_create_missing_path(self, tmp_path: Path) -> None:
        """Test create with missing path."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="path is required"):
            await tool(command="create", file_text="Content")

    @pytest.mark.asyncio
    async def test_create_missing_file_text(self, tmp_path: Path) -> None:
        """Test create with missing file_text."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="file_text is required"):
            await tool(command="create", path="/memories/test.txt")


class TestClaudeMemoryStrReplace:
    """Test str_replace command."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="str_replace uses shell commands not available on Windows",
    )
    async def test_str_replace(self, tmp_path: Path) -> None:
        """Test replacing text in a file."""
        file_path = tmp_path / "replace_test.txt"
        file_path.write_text("Hello, World!")

        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        await tool(
            command="str_replace",
            path="/memories/replace_test.txt",
            old_str="World",
            new_str="Memory",
        )

        # Verify replacement (write_file_async uses heredoc which adds trailing newline)
        assert file_path.read_text().rstrip("\n") == "Hello, Memory!"

    @pytest.mark.asyncio
    async def test_str_replace_missing_path(self, tmp_path: Path) -> None:
        """Test str_replace with missing path."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="path is required"):
            await tool(command="str_replace", old_str="old", new_str="new")

    @pytest.mark.asyncio
    async def test_str_replace_nonexistent_file(self, tmp_path: Path) -> None:
        """Test str_replace on nonexistent file."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="does not exist"):
            await tool(
                command="str_replace",
                path="/memories/nonexistent.txt",
                old_str="old",
                new_str="new",
            )


class TestClaudeMemoryDelete:
    """Test delete command."""

    @pytest.mark.asyncio
    async def test_delete_file(self, tmp_path: Path) -> None:
        """Test deleting a file."""
        file_path = tmp_path / "to_delete.txt"
        file_path.write_text("Delete me")

        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        result = await tool(command="delete", path="/memories/to_delete.txt")

        assert isinstance(result[0], TextContent)
        assert "Successfully deleted" in result[0].text
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_delete_directory(self, tmp_path: Path) -> None:
        """Test deleting a directory."""
        dir_path = tmp_path / "to_delete"
        dir_path.mkdir()
        (dir_path / "file.txt").write_text("Content")

        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        result = await tool(command="delete", path="/memories/to_delete")

        assert isinstance(result[0], TextContent)
        assert "Successfully deleted" in result[0].text
        assert not dir_path.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Test deleting nonexistent path."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="does not exist"):
            await tool(command="delete", path="/memories/nonexistent")


class TestClaudeMemoryRename:
    """Test rename command."""

    @pytest.mark.asyncio
    async def test_rename_file(self, tmp_path: Path) -> None:
        """Test renaming a file."""
        old_path = tmp_path / "old_name.txt"
        old_path.write_text("Content")

        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))
        result = await tool(
            command="rename",
            old_path="/memories/old_name.txt",
            new_path="/memories/new_name.txt",
        )

        assert isinstance(result[0], TextContent)
        assert "Successfully renamed" in result[0].text
        assert not old_path.exists()
        assert (tmp_path / "new_name.txt").exists()

    @pytest.mark.asyncio
    async def test_rename_nonexistent(self, tmp_path: Path) -> None:
        """Test renaming nonexistent file."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="does not exist"):
            await tool(
                command="rename",
                old_path="/memories/nonexistent.txt",
                new_path="/memories/new.txt",
            )

    @pytest.mark.asyncio
    async def test_rename_destination_exists(self, tmp_path: Path) -> None:
        """Test renaming to existing destination."""
        (tmp_path / "source.txt").write_text("Source")
        (tmp_path / "dest.txt").write_text("Dest")

        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="already exists"):
            await tool(
                command="rename",
                old_path="/memories/source.txt",
                new_path="/memories/dest.txt",
            )


class TestClaudeMemoryInvalidCommand:
    """Test invalid command handling."""

    @pytest.mark.asyncio
    async def test_invalid_command(self, tmp_path: Path) -> None:
        """Test handling of invalid command."""
        tool = ClaudeMemoryTool(memories_dir=str(tmp_path))

        with pytest.raises(ToolError, match="Unrecognized command"):
            await tool(command="invalid_command")  # type: ignore[arg-type]


class TestClaudeMemoryCommand:
    """Tests for ClaudeMemoryCommand type."""

    def test_command_variants(self) -> None:
        """Test all command variants are defined."""
        commands = get_args(ClaudeMemoryCommand)
        assert "view" in commands
        assert "create" in commands
        assert "str_replace" in commands
        assert "insert" in commands
        assert "delete" in commands
        assert "rename" in commands

    def test_command_count(self) -> None:
        """Test expected number of commands."""
        commands = get_args(ClaudeMemoryCommand)
        assert len(commands) == 6
