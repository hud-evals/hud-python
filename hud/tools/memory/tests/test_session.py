"""Tests for session memory tool."""

from __future__ import annotations

import pytest
from mcp.types import TextContent

from hud.tools.memory import SessionMemoryTool
from hud.tools.memory.base import BaseSessionMemoryTool


def test_session_memory_add_and_query() -> None:
    """Test adding and querying session memory."""
    store = SessionMemoryTool()
    store.add_entry("apple orange", {"kind": "fruit"})
    store.add_entry("carrot celery", {"kind": "veg"})

    results = store.search_entries("apple", top_k=5)
    assert len(results) == 1
    assert results[0].metadata["kind"] == "fruit"


@pytest.mark.asyncio
async def test_session_memory_tool_add_and_search() -> None:
    """Test SessionMemoryTool add and search actions."""
    tool = SessionMemoryTool()

    out_add = await tool(action="add", text="alpha beta", metadata={"id": 1})
    assert isinstance(out_add[0], TextContent)

    out_search = await tool(action="search", text="alpha")
    assert isinstance(out_search[0], TextContent)
    assert out_search[0].text.startswith("1.")


@pytest.mark.asyncio
async def test_session_memory_tool_unknown_action() -> None:
    """Test SessionMemoryTool with unknown action."""
    tool = SessionMemoryTool()
    res = await tool(action="noop", text="x")
    assert isinstance(res[0], TextContent)
    assert res[0].text == "unknown action"


def test_tokenize() -> None:
    """Test tokenization utility."""
    tokens = BaseSessionMemoryTool.tokenize("Hello World")
    assert tokens == {"hello", "world"}


def test_jaccard_similarity() -> None:
    """Test Jaccard similarity calculation."""
    a = {"hello", "world"}
    b = {"hello", "there"}
    similarity = BaseSessionMemoryTool.jaccard_similarity(a, b)
    # Intersection = 1 (hello), Union = 3 (hello, world, there)
    assert similarity == pytest.approx(1 / 3)
