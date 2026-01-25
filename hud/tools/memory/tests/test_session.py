"""Tests for session memory tool."""

from __future__ import annotations

from typing import Any

import pytest
from mcp.types import TextContent

from hud.tools.memory import SessionMemoryTool
from hud.tools.memory.base import BaseSessionMemoryTool, MemoryEntry


class TestBaseSessionMemoryTool:
    """Tests for BaseSessionMemoryTool base class."""

    def test_tokenize_basic(self) -> None:
        """Test basic tokenization."""
        tokens = BaseSessionMemoryTool.tokenize("Hello World")
        assert tokens == {"hello", "world"}

    def test_tokenize_with_punctuation(self) -> None:
        """Test tokenization with punctuation (keeps punctuation attached)."""
        tokens = BaseSessionMemoryTool.tokenize("Hello, World! How are you?")
        # Tokenize doesn't strip punctuation, just lowercases
        assert "hello," in tokens
        assert "world!" in tokens
        assert "how" in tokens
        assert "are" in tokens
        assert "you?" in tokens

    def test_tokenize_empty_string(self) -> None:
        """Test tokenization of empty string."""
        tokens = BaseSessionMemoryTool.tokenize("")
        assert tokens == set()

    def test_tokenize_with_numbers(self) -> None:
        """Test tokenization handles numbers."""
        tokens = BaseSessionMemoryTool.tokenize("test 123 foo")
        assert "test" in tokens
        assert "123" in tokens
        assert "foo" in tokens

    def test_jaccard_similarity_identical(self) -> None:
        """Test Jaccard similarity for identical sets."""
        a = {"hello", "world"}
        similarity = BaseSessionMemoryTool.jaccard_similarity(a, a)
        assert similarity == 1.0

    def test_jaccard_similarity_disjoint(self) -> None:
        """Test Jaccard similarity for disjoint sets."""
        a = {"hello", "world"}
        b = {"foo", "bar"}
        similarity = BaseSessionMemoryTool.jaccard_similarity(a, b)
        assert similarity == 0.0

    def test_jaccard_similarity_partial(self) -> None:
        """Test Jaccard similarity for partial overlap."""
        a = {"hello", "world"}
        b = {"hello", "there"}
        similarity = BaseSessionMemoryTool.jaccard_similarity(a, b)
        # Intersection = 1 (hello), Union = 3 (hello, world, there)
        assert similarity == pytest.approx(1 / 3)

    def test_jaccard_similarity_empty_sets(self) -> None:
        """Test Jaccard similarity for empty sets."""
        a: set[str] = set()
        b: set[str] = set()
        similarity = BaseSessionMemoryTool.jaccard_similarity(a, b)
        assert similarity == 0.0


class TestSessionMemoryToolInit:
    """Tests for SessionMemoryTool initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        tool = SessionMemoryTool()
        assert tool.name == "memory"
        assert tool.title == "Memory"
        assert "session memory" in tool.description.lower()

    def test_custom_collection(self) -> None:
        """Test initialization with custom collection name."""
        tool = SessionMemoryTool(collection="custom_collection")
        # Should not raise, backend defaults to self
        assert tool._backend is tool

    def test_parameters_schema(self) -> None:
        """Test parameters property returns valid schema."""
        tool = SessionMemoryTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "text" in params["properties"]
        assert "metadata" in params["properties"]
        assert "top_k" in params["properties"]
        assert params["required"] == ["action", "text"]


class TestSessionMemoryToolAddSearch:
    """Tests for add and search functionality."""

    def test_add_and_query(self) -> None:
        """Test adding and querying session memory."""
        store = SessionMemoryTool()
        store.add_entry("apple orange", {"kind": "fruit"})
        store.add_entry("carrot celery", {"kind": "veg"})

        results = store.search_entries("apple", top_k=5)
        assert len(results) == 1
        assert results[0].metadata["kind"] == "fruit"

    def test_search_no_matches(self) -> None:
        """Test search with no matches."""
        store = SessionMemoryTool()
        store.add_entry("apple orange", {"kind": "fruit"})

        results = store.search_entries("zebra", top_k=5)
        assert len(results) == 0

    def test_search_top_k_limit(self) -> None:
        """Test search respects top_k limit."""
        store = SessionMemoryTool()
        for i in range(10):
            store.add_entry(f"item {i} test", {"id": i})

        results = store.search_entries("test", top_k=3)
        assert len(results) == 3

    def test_add_without_metadata(self) -> None:
        """Test adding entry without metadata."""
        store = SessionMemoryTool()
        store.add_entry("simple text")

        results = store.search_entries("simple", top_k=5)
        assert len(results) == 1
        assert results[0].metadata is None or results[0].metadata == {}

    @pytest.mark.asyncio
    async def test_tool_add_action(self) -> None:
        """Test add action via tool call."""
        tool = SessionMemoryTool()

        result = await tool(action="add", text="alpha beta", metadata={"id": 1})
        assert isinstance(result[0], TextContent)
        assert result[0].text == "stored"

    @pytest.mark.asyncio
    async def test_tool_search_action(self) -> None:
        """Test search action via tool call."""
        tool = SessionMemoryTool()

        await tool(action="add", text="alpha beta gamma")
        result = await tool(action="search", text="alpha")

        assert isinstance(result[0], TextContent)
        assert "alpha beta gamma" in result[0].text

    @pytest.mark.asyncio
    async def test_tool_search_no_matches(self) -> None:
        """Test search action with no matches."""
        tool = SessionMemoryTool()

        result = await tool(action="search", text="nonexistent")
        assert isinstance(result[0], TextContent)
        assert result[0].text == "no matches"

    @pytest.mark.asyncio
    async def test_tool_search_with_metadata(self) -> None:
        """Test search returns metadata in results."""
        tool = SessionMemoryTool()

        await tool(action="add", text="test item", metadata={"category": "demo"})
        result = await tool(action="search", text="test")

        assert isinstance(result[0], TextContent)
        assert "category" in result[0].text or "demo" in result[0].text

    @pytest.mark.asyncio
    async def test_tool_search_multiple_results(self) -> None:
        """Test search with multiple results."""
        tool = SessionMemoryTool()

        await tool(action="add", text="python programming language")
        await tool(action="add", text="javascript programming language")
        await tool(action="add", text="rust programming language")

        result = await tool(action="search", text="programming")
        text = result[0].text

        assert "1." in text
        assert "2." in text
        assert "3." in text

    @pytest.mark.asyncio
    async def test_tool_search_custom_top_k(self) -> None:
        """Test search with custom top_k."""
        tool = SessionMemoryTool()

        for i in range(10):
            await tool(action="add", text=f"item number {i}")

        result = await tool(action="search", text="item", top_k=2)
        lines = [line for line in result[0].text.split("\n") if line.strip()]
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_tool_unknown_action(self) -> None:
        """Test unknown action returns error message."""
        tool = SessionMemoryTool()

        result = await tool(action="invalid", text="test")
        assert isinstance(result[0], TextContent)
        assert result[0].text == "unknown action"


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_memory_entry_creation(self) -> None:
        """Test creating a MemoryEntry."""
        entry = MemoryEntry(text="test", metadata={"key": "value"}, tokens={"test"})
        assert entry.text == "test"
        assert entry.metadata == {"key": "value"}
        assert entry.tokens == {"test"}

    def test_memory_entry_empty_metadata(self) -> None:
        """Test MemoryEntry with empty metadata."""
        entry = MemoryEntry(text="test", metadata={}, tokens={"test"})
        assert entry.text == "test"
        assert entry.metadata == {}
        assert entry.tokens == {"test"}


class TestBackwardsCompatibility:
    """Tests for backwards compatibility."""

    def test_memory_tool_alias(self) -> None:
        """Test MemoryTool alias for SessionMemoryTool."""
        from hud.tools.memory.session import MemoryTool

        assert MemoryTool is SessionMemoryTool

    def test_session_memory_tool_in_exports(self) -> None:
        """Test SessionMemoryTool is exported from module."""
        from hud.tools.memory import SessionMemoryTool as ST

        assert ST is SessionMemoryTool
