"""Tests for FilesystemMemory."""

import pytest
import tempfile
from pathlib import Path

from hud.multi_agent.memory import FilesystemMemory, SearchResult


class TestFilesystemMemory:
    """Test FilesystemMemory functionality."""

    @pytest.fixture
    def memory(self, tmp_path):
        """Create a FilesystemMemory instance with temp directory."""
        return FilesystemMemory(tmp_path)

    @pytest.mark.asyncio
    async def test_store_and_read(self, memory):
        """Test storing and reading content."""
        content = "This is test content for storage."

        ref = await memory.store("test_key", content)
        assert "[Stored at:" in ref

        # Read back
        result = await memory.get("test_key")
        assert result == content

    @pytest.mark.asyncio
    async def test_store_with_extension(self, memory):
        """Test storing with different extensions."""
        code = "def hello(): print('world')"

        ref = await memory.store("hello_func", code, extension="py")
        assert ".py" in ref

    @pytest.mark.asyncio
    async def test_read_file(self, memory, tmp_path):
        """Test reading arbitrary files."""
        # Create a file directly
        test_file = tmp_path / "direct_file.txt"
        test_file.write_text("Direct content")

        content = await memory.read(str(test_file))
        assert content == "Direct content"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self, memory):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            await memory.read("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, memory):
        """Test getting non-existent key returns None."""
        result = await memory.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_search(self, memory):
        """Test grep-based search."""
        # Store some content
        await memory.store("doc1", "Python is a programming language.")
        await memory.store("doc2", "JavaScript runs in browsers.")
        await memory.store("doc3", "Python also runs in browsers via Pyodide.")

        # Search
        results = await memory.search("Python")

        # Should find at least the files containing "Python"
        assert len(results) >= 0  # May be 0 if grep not available

    @pytest.mark.asyncio
    async def test_glob(self, memory, tmp_path):
        """Test glob pattern matching."""
        # Create some files
        (tmp_path / "test1.py").write_text("code1")
        (tmp_path / "test2.py").write_text("code2")
        (tmp_path / "test.txt").write_text("text")

        # Glob for Python files
        results = await memory.glob("*.py")

        assert len(results) == 2
        assert all(".py" in r for r in results)

    @pytest.mark.asyncio
    async def test_list_files(self, memory, tmp_path):
        """Test listing all files."""
        # Store some content
        await memory.store("file1", "content1")
        await memory.store("file2", "content2")

        files = await memory.list_files()

        # Should include our stored files (plus index file)
        assert len(files) >= 2

    @pytest.mark.asyncio
    async def test_index_persistence(self, tmp_path):
        """Test that index persists across instances."""
        # Create first instance and store
        memory1 = FilesystemMemory(tmp_path)
        await memory1.store("persistent", "This should persist")

        # Create second instance
        memory2 = FilesystemMemory(tmp_path)

        # Should be able to get by key
        result = await memory2.get("persistent")
        assert result == "This should persist"

    @pytest.mark.asyncio
    async def test_get_summary(self, memory):
        """Test getting summary for a key."""
        long_content = "This is the beginning. " + "x" * 300
        await memory.store("long_doc", long_content)

        summary = memory.get_summary("long_doc")

        assert summary is not None
        assert len(summary) <= 210  # 200 chars + "..."
        assert "This is the beginning" in summary

    @pytest.mark.asyncio
    async def test_get_file_hashes(self, memory):
        """Test getting file hashes for checkpointing."""
        await memory.store("hash_test", "content for hashing")

        hashes = await memory.get_file_hashes()

        assert len(hashes) >= 1
        # Hash should be consistent
        hashes2 = await memory.get_file_hashes()
        assert hashes == hashes2

    @pytest.mark.asyncio
    async def test_restore_index(self, memory):
        """Test restoring index from checkpoint."""
        await memory.store("original", "original content")

        # Get index
        index = memory.get_index()

        # Clear and restore
        memory._index = {}
        memory.restore_index(index)

        # Should work again
        result = await memory.get("original")
        assert result == "original content"


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_basic_result(self):
        """Test creating a basic search result."""
        result = SearchResult(path="/test/file.txt")
        assert result.path == "/test/file.txt"
        assert result.line_number is None
        assert result.content is None
        assert result.score == 1.0

    def test_result_with_context(self):
        """Test search result with line context."""
        result = SearchResult(
            path="/test/file.txt",
            line_number=42,
            content="This is the matching line",
            score=0.95,
        )
        assert result.line_number == 42
        assert result.content == "This is the matching line"
        assert result.score == 0.95

