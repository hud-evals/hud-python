"""Tests for apply_patch tool."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from hud.tools.apply_patch import (
    ActionType,
    ApplyPatchResult,
    ApplyPatchTool,
    Chunk,
    Commit,
    DiffError,
    FileChange,
    Parser,
    Patch,
    PatchAction,
    _apply_commit,
    _find_context,
    _find_context_core,
    _get_updated_file,
    _identify_files_needed,
    _patch_to_commit,
    _text_to_patch,
)


class TestApplyPatchResult:
    """Tests for ApplyPatchResult dataclass."""

    def test_to_dict_completed(self):
        """Test to_dict for completed result."""
        result = ApplyPatchResult(status="completed", output="Success")
        assert result.to_dict() == {"status": "completed", "output": "Success"}

    def test_to_dict_failed(self):
        """Test to_dict for failed result."""
        result = ApplyPatchResult(status="failed", output="Error message")
        assert result.to_dict() == {"status": "failed", "output": "Error message"}


class TestParser:
    """Tests for Parser class."""

    def test_is_done_at_end(self):
        """Test is_done when at end of lines."""
        parser = Parser(current_files={}, lines=["line1"], index=1)
        assert parser.is_done() is True

    def test_is_done_with_prefix(self):
        """Test is_done with matching prefix."""
        parser = Parser(current_files={}, lines=["*** End Patch"], index=0)
        assert parser.is_done(("*** End Patch",)) is True

    def test_is_done_no_match(self):
        """Test is_done when prefix doesn't match."""
        parser = Parser(current_files={}, lines=["other line"], index=0)
        assert parser.is_done(("*** End Patch",)) is False

    def test_startswith(self):
        """Test startswith method."""
        parser = Parser(current_files={}, lines=["*** Update File: test.txt"], index=0)
        assert parser.startswith("*** Update File:") is True
        assert parser.startswith("*** Delete File:") is False

    def test_read_str_with_prefix(self):
        """Test read_str extracts text after prefix."""
        parser = Parser(current_files={}, lines=["*** Update File: test.txt"], index=0)
        result = parser.read_str("*** Update File: ")
        assert result == "test.txt"
        assert parser.index == 1

    def test_read_str_no_match(self):
        """Test read_str returns empty when prefix doesn't match."""
        parser = Parser(current_files={}, lines=["other line"], index=0)
        result = parser.read_str("*** Update File: ")
        assert result == ""
        assert parser.index == 0

    def test_read_str_return_everything(self):
        """Test read_str with return_everything=True."""
        parser = Parser(current_files={}, lines=["*** Update File: test.txt"], index=0)
        result = parser.read_str("*** Update File: ", return_everything=True)
        assert result == "*** Update File: test.txt"

    def test_parse_add_file(self):
        """Test parsing add file action."""
        lines = [
            "*** Begin Patch",
            "*** Add File: new.txt",
            "+line 1",
            "+line 2",
            "*** End Patch",
        ]
        parser = Parser(current_files={}, lines=lines, index=1)
        parser.parse()

        assert "new.txt" in parser.patch.actions
        action = parser.patch.actions["new.txt"]
        assert action.type == ActionType.ADD
        assert action.new_file == "line 1\nline 2"

    def test_parse_delete_file(self):
        """Test parsing delete file action."""
        lines = [
            "*** Begin Patch",
            "*** Delete File: old.txt",
            "*** End Patch",
        ]
        parser = Parser(current_files={"old.txt": "content"}, lines=lines, index=1)
        parser.parse()

        assert "old.txt" in parser.patch.actions
        action = parser.patch.actions["old.txt"]
        assert action.type == ActionType.DELETE

    def test_parse_missing_end_patch(self):
        """Test that truncated patch (no end marker) raises error."""
        lines = [
            "*** Begin Patch",
            "*** Add File: new.txt",
            "+content",
        ]
        parser = Parser(current_files={}, lines=lines, index=1)
        with pytest.raises(DiffError, match="Missing End Patch"):
            parser.parse()

    def test_parse_truncated_update_file(self):
        """Test that truncated update file patch raises DiffError, not AssertionError."""
        lines = [
            "*** Begin Patch",
            "*** Update File: test.txt",
        ]
        parser = Parser(current_files={"test.txt": "content"}, lines=lines, index=1)
        # Should raise DiffError for unexpected EOF, not AssertionError
        with pytest.raises(DiffError):
            parser.parse()

    def test_startswith_at_eof(self):
        """Test that startswith at EOF raises DiffError, not AssertionError."""
        parser = Parser(current_files={}, lines=["line"], index=1)  # index past end
        with pytest.raises(DiffError, match="Unexpected end of patch"):
            parser.startswith("test")

    def test_read_str_at_eof(self):
        """Test that read_str at EOF returns empty string, not AssertionError."""
        parser = Parser(current_files={}, lines=["line"], index=1)  # index past end
        result = parser.read_str("test")
        assert result == ""

    def test_parse_wrong_end_marker(self):
        """Test that wrong end marker in add file content raises error."""
        lines = [
            "*** Begin Patch",
            "*** Add File: new.txt",
            "+content",
            "*** Wrong End",  # This is inside the add file, so it's an invalid line
        ]
        parser = Parser(current_files={}, lines=lines, index=1)
        with pytest.raises(DiffError, match="Invalid Add File Line"):
            parser.parse()

    def test_parse_duplicate_path_error(self):
        """Test that duplicate paths raise error."""
        lines = [
            "*** Begin Patch",
            "*** Add File: test.txt",
            "+content",
            "*** Add File: test.txt",
            "+more content",
            "*** End Patch",
        ]
        parser = Parser(current_files={}, lines=lines, index=1)
        with pytest.raises(DiffError, match="Duplicate Path"):
            parser.parse()

    def test_parse_update_missing_file_error(self):
        """Test that updating missing file raises error."""
        lines = [
            "*** Begin Patch",
            "*** Update File: nonexistent.txt",
            " context",
            "*** End Patch",
        ]
        parser = Parser(current_files={}, lines=lines, index=1)
        with pytest.raises(DiffError, match="Missing File"):
            parser.parse()

    def test_parse_delete_missing_file_error(self):
        """Test that deleting missing file raises error."""
        lines = [
            "*** Begin Patch",
            "*** Delete File: nonexistent.txt",
            "*** End Patch",
        ]
        parser = Parser(current_files={}, lines=lines, index=1)
        with pytest.raises(DiffError, match="Missing File"):
            parser.parse()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_find_context_core_exact_match(self):
        """Test _find_context_core with exact match."""
        lines = ["a", "b", "c", "d"]
        context = ["b", "c"]
        index, fuzz = _find_context_core(lines, context, 0)
        assert index == 1
        assert fuzz == 0

    def test_find_context_core_rstrip_match(self):
        """Test _find_context_core with rstrip match."""
        lines = ["a", "b  ", "c  ", "d"]
        context = ["b", "c"]
        index, fuzz = _find_context_core(lines, context, 0)
        assert index == 1
        assert fuzz == 1

    def test_find_context_core_strip_match(self):
        """Test _find_context_core with strip match."""
        lines = ["a", "  b  ", "  c  ", "d"]
        context = ["b", "c"]
        index, fuzz = _find_context_core(lines, context, 0)
        assert index == 1
        assert fuzz == 100

    def test_find_context_core_no_match(self):
        """Test _find_context_core with no match."""
        lines = ["a", "b", "c"]
        context = ["x", "y"]
        index, _ = _find_context_core(lines, context, 0)
        assert index == -1

    def test_find_context_core_empty_context(self):
        """Test _find_context_core with empty context."""
        lines = ["a", "b"]
        index, fuzz = _find_context_core(lines, [], 0)
        assert index == 0
        assert fuzz == 0

    def test_find_context_eof(self):
        """Test _find_context with EOF flag."""
        lines = ["a", "b", "c", "d"]
        context = ["c", "d"]
        index, fuzz = _find_context(lines, context, 0, eof=True)
        assert index == 2
        assert fuzz == 0

    def test_identify_files_needed(self):
        """Test _identify_files_needed."""
        text = """*** Begin Patch
*** Update File: file1.txt
 context
*** Delete File: file2.txt
*** Add File: file3.txt
+new content
*** End Patch"""
        files = _identify_files_needed(text)
        assert set(files) == {"file1.txt", "file2.txt"}

    def test_get_updated_file_simple(self):
        """Test _get_updated_file with simple update."""
        text = "line1\nline2\nline3"
        action = PatchAction(
            type=ActionType.UPDATE,
            chunks=[
                Chunk(orig_index=1, del_lines=["line2"], ins_lines=["new line2"]),
            ],
        )
        result = _get_updated_file(text, action, "test.txt")
        assert result == "line1\nnew line2\nline3"

    def test_patch_to_commit_add(self):
        """Test _patch_to_commit with add action."""
        patch = Patch(actions={"new.txt": PatchAction(type=ActionType.ADD, new_file="content")})
        commit = _patch_to_commit(patch, {})
        assert "new.txt" in commit.changes
        assert commit.changes["new.txt"].type == ActionType.ADD
        assert commit.changes["new.txt"].new_content == "content"

    def test_patch_to_commit_delete(self):
        """Test _patch_to_commit with delete action."""
        patch = Patch(actions={"old.txt": PatchAction(type=ActionType.DELETE)})
        orig = {"old.txt": "old content"}
        commit = _patch_to_commit(patch, orig)
        assert commit.changes["old.txt"].type == ActionType.DELETE
        assert commit.changes["old.txt"].old_content == "old content"

    def test_apply_commit(self):
        """Test _apply_commit function."""
        written = {}
        removed = []

        def write_fn(path, content):
            written[path] = content

        def remove_fn(path):
            removed.append(path)

        commit = Commit(
            changes={
                "new.txt": FileChange(type=ActionType.ADD, new_content="new content"),
                "old.txt": FileChange(type=ActionType.DELETE, old_content="old"),
            }
        )
        _apply_commit(commit, write_fn, remove_fn)

        assert written == {"new.txt": "new content"}
        assert removed == ["old.txt"]

    def test_apply_commit_with_move(self):
        """Test _apply_commit with move operation."""
        written = {}
        removed = []

        def write_fn(path, content):
            written[path] = content

        def remove_fn(path):
            removed.append(path)

        commit = Commit(
            changes={
                "old.txt": FileChange(
                    type=ActionType.UPDATE,
                    old_content="old",
                    new_content="new",
                    move_path="renamed.txt",
                ),
            }
        )
        _apply_commit(commit, write_fn, remove_fn)

        assert written == {"renamed.txt": "new"}
        assert removed == ["old.txt"]

    def test_text_to_patch_invalid(self):
        """Test _text_to_patch with invalid patch text."""
        with pytest.raises(DiffError, match="Invalid patch text"):
            _text_to_patch("invalid", {})

    def test_text_to_patch_valid(self):
        """Test _text_to_patch with valid patch."""
        text = """*** Begin Patch
*** Add File: test.txt
+content
*** End Patch"""
        patch, fuzz = _text_to_patch(text, {})
        assert "test.txt" in patch.actions
        assert fuzz == 0


class TestApplyPatchTool:
    """Tests for ApplyPatchTool."""

    def test_init_default(self):
        """Test default initialization."""
        tool = ApplyPatchTool()
        assert tool.base_path == os.path.abspath(".")

    def test_init_with_base_path(self):
        """Test initialization with custom base path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            assert tool.base_path == os.path.abspath(tmpdir)

    def test_validate_path_absolute(self):
        """Test that absolute paths are rejected."""
        tool = ApplyPatchTool()
        with pytest.raises(DiffError, match="Absolute paths are not allowed"):
            tool._validate_path("/absolute/path")

    def test_validate_path_traversal(self):
        """Test that path traversal is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            with pytest.raises(DiffError, match="Path traversal detected"):
                tool._validate_path("../outside")

    def test_validate_path_traversal_sibling_prefix(self):
        """Test that path traversal via sibling directory with shared prefix is detected.

        Bug: Path traversal check bypassed via sibling directory prefix.

        The path traversal check `full_path.startswith(self.base_path)` uses string
        prefix matching, which can be bypassed when sibling directories share a name
        prefix with the base directory. For example, if base_path is /tmp/myapp and
        a user provides path ../myapp_sibling/secret.txt, the resolved full_path
        becomes /tmp/myapp_sibling/secret.txt. The check passes because the string
        /tmp/myapp_sibling/secret.txt starts with /tmp/myapp, allowing access to
        files outside the intended sandbox.

        The fix is to ensure a path separator follows the base path
        (e.g., full_path.startswith(self.base_path + os.sep)) or use os.path.commonpath.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base directory "myapp" and sibling directory "myapp_sibling"
            base_dir = os.path.join(tmpdir, "myapp")
            sibling_dir = os.path.join(tmpdir, "myapp_sibling")
            os.makedirs(base_dir)
            os.makedirs(sibling_dir)

            # Create a "secret" file in the sibling directory
            secret_file = os.path.join(sibling_dir, "secret.txt")
            Path(secret_file).write_text("secret content")

            tool = ApplyPatchTool(base_path=base_dir)

            # Attempt to access the sibling directory via path traversal
            # This should be detected as path traversal, but the bug allows it
            # because "/tmp/.../myapp_sibling/secret.txt".startswith("/tmp/.../myapp")
            # returns True due to string prefix matching
            with pytest.raises(DiffError, match="Path traversal detected"):
                tool._validate_path("../myapp_sibling/secret.txt")

    def test_validate_path_valid(self):
        """Test valid path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            result = tool._validate_path("subdir/file.txt")
            assert result == os.path.join(tmpdir, "subdir/file.txt")

    @pytest.mark.asyncio
    async def test_call_missing_type(self):
        """Test call with missing operation type."""
        tool = ApplyPatchTool()
        result = await tool(path="test.txt")
        assert result.status == "failed"
        assert "Missing operation type" in result.output

    @pytest.mark.asyncio
    async def test_call_missing_path(self):
        """Test call with missing path."""
        tool = ApplyPatchTool()
        result = await tool(type="create_file")
        assert result.status == "failed"
        assert "Missing file path" in result.output

    @pytest.mark.asyncio
    async def test_call_unknown_type(self):
        """Test call with unknown operation type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(type="unknown_op", path="test.txt")
            assert result.status == "failed"
            assert "Unknown operation type" in result.output

    @pytest.mark.asyncio
    async def test_create_file_success(self):
        """Test successful file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="create_file",
                path="new.txt",
                diff="+line 1\n+line 2",
            )
            assert result.status == "completed"
            assert "Created" in result.output

            # Verify file was created
            with open(os.path.join(tmpdir, "new.txt")) as f:  # noqa: ASYNC230
                content = f.read()
            assert content == "line 1\nline 2"

    @pytest.mark.asyncio
    async def test_create_file_already_exists(self):
        """Test creating file that already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing file
            existing_path = os.path.join(tmpdir, "existing.txt")
            Path(existing_path).write_text("existing content")

            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="create_file",
                path="existing.txt",
                diff="+new content",
            )
            assert result.status == "failed"
            assert "already exists" in result.output

    @pytest.mark.asyncio
    async def test_create_file_missing_diff(self):
        """Test creating file without diff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="create_file",
                path="new.txt",
            )
            assert result.status == "failed"
            assert "Missing diff" in result.output

    @pytest.mark.asyncio
    async def test_delete_file_success(self):
        """Test successful file deletion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file to delete
            file_path = os.path.join(tmpdir, "to_delete.txt")
            Path(file_path).write_text("content")

            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="delete_file",
                path="to_delete.txt",
            )
            assert result.status == "completed"
            assert "Deleted" in result.output
            assert not os.path.exists(file_path)

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self):
        """Test deleting non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="delete_file",
                path="nonexistent.txt",
            )
            assert result.status == "failed"
            assert "not found" in result.output

    @pytest.mark.asyncio
    async def test_update_file_success(self):
        """Test successful file update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file to update
            file_path = os.path.join(tmpdir, "test.txt")
            Path(file_path).write_text("line1\nline2\nline3")

            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="update_file",
                path="test.txt",
                diff=" line1\n-line2\n+new line2\n line3",
            )
            assert result.status == "completed"
            assert "Updated" in result.output

            # Verify file was updated
            with open(file_path) as f:  # noqa: ASYNC230
                content = f.read()
            assert content == "line1\nnew line2\nline3"

    @pytest.mark.asyncio
    async def test_update_file_not_found(self):
        """Test updating non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="update_file",
                path="nonexistent.txt",
                diff=" line1\n-line2\n+new line2",
            )
            assert result.status == "failed"
            assert "not found" in result.output

    @pytest.mark.asyncio
    async def test_update_file_missing_diff(self):
        """Test updating file without diff."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file
            file_path = os.path.join(tmpdir, "test.txt")
            Path(file_path).write_text("content")

            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="update_file",
                path="test.txt",
            )
            assert result.status == "failed"
            assert "Missing diff" in result.output

    @pytest.mark.asyncio
    async def test_create_file_with_subdirectory(self):
        """Test creating file in subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="create_file",
                path="subdir/nested/file.txt",
                diff="+content",
            )
            assert result.status == "completed"

            # Verify file was created in subdirectory
            file_path = os.path.join(tmpdir, "subdir/nested/file.txt")
            assert os.path.exists(file_path)
            with open(file_path) as f:  # noqa: ASYNC230
                assert f.read() == "content"

    def test_parse_create_diff(self):
        """Test _parse_create_diff method."""
        tool = ApplyPatchTool()
        content = tool._parse_create_diff("+line 1\n+line 2\n+line 3")
        assert content == "line 1\nline 2\nline 3"

    def test_parse_create_diff_with_spaces(self):
        """Test _parse_create_diff with space-prefixed lines."""
        tool = ApplyPatchTool()
        content = tool._parse_create_diff("+line 1\n context\n+line 3")
        assert content == "line 1\ncontext\nline 3"

    def test_open_file_not_found(self):
        """Test _open_file with non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            with pytest.raises(DiffError, match="File not found"):
                tool._open_file("nonexistent.txt")

    def test_write_file(self):
        """Test _write_file method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ApplyPatchTool(base_path=tmpdir)
            tool._write_file("test.txt", "content")

            with open(os.path.join(tmpdir, "test.txt")) as f:
                assert f.read() == "content"

    def test_remove_file(self):
        """Test _remove_file method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file
            file_path = os.path.join(tmpdir, "test.txt")
            Path(file_path).write_text("content")

            tool = ApplyPatchTool(base_path=tmpdir)
            tool._remove_file("test.txt")

            assert not os.path.exists(file_path)

    def test_load_files(self):
        """Test _load_files method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            Path(os.path.join(tmpdir, "file1.txt")).write_text("content1")
            Path(os.path.join(tmpdir, "file2.txt")).write_text("content2")

            tool = ApplyPatchTool(base_path=tmpdir)
            files = tool._load_files(["file1.txt", "file2.txt"])

            assert files == {"file1.txt": "content1", "file2.txt": "content2"}

    @pytest.mark.asyncio
    async def test_update_with_fuzz(self):
        """Test update that requires fuzzy matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with trailing whitespace
            file_path = os.path.join(tmpdir, "test.txt")
            Path(file_path).write_text("line1  \nline2\nline3")

            tool = ApplyPatchTool(base_path=tmpdir)
            result = await tool(
                type="update_file",
                path="test.txt",
                diff=" line1\n-line2\n+new line2\n line3",
            )
            assert result.status == "completed"
            # Fuzz > 0 should be reported
            assert "Updated" in result.output


class TestDataclasses:
    """Tests for dataclass structures."""

    def test_file_change(self):
        """Test FileChange dataclass."""
        change = FileChange(
            type=ActionType.UPDATE,
            old_content="old",
            new_content="new",
            move_path="moved.txt",
        )
        assert change.type == ActionType.UPDATE
        assert change.old_content == "old"
        assert change.new_content == "new"
        assert change.move_path == "moved.txt"

    def test_commit(self):
        """Test Commit dataclass."""
        commit = Commit()
        assert commit.changes == {}
        commit.changes["test.txt"] = FileChange(type=ActionType.ADD, new_content="content")
        assert "test.txt" in commit.changes

    def test_chunk(self):
        """Test Chunk dataclass."""
        chunk = Chunk(orig_index=5, del_lines=["old"], ins_lines=["new"])
        assert chunk.orig_index == 5
        assert chunk.del_lines == ["old"]
        assert chunk.ins_lines == ["new"]

    def test_patch_action(self):
        """Test PatchAction dataclass."""
        action = PatchAction(type=ActionType.ADD, new_file="content")
        assert action.type == ActionType.ADD
        assert action.new_file == "content"
        assert action.chunks == []
        assert action.move_path is None

    def test_patch(self):
        """Test Patch dataclass."""
        patch = Patch()
        assert patch.actions == {}

    def test_action_type_enum(self):
        """Test ActionType enum values."""
        assert ActionType.ADD.value == "add"
        assert ActionType.DELETE.value == "delete"
        assert ActionType.UPDATE.value == "update"
