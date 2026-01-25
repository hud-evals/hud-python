from __future__ import annotations

import pytest
from mcp.types import TextContent

from hud.tools.memory import InMemoryStore, MemoryTool


def test_inmemory_store_add_and_query() -> None:
    store = InMemoryStore()
    store.add("apple orange", {"kind": "fruit"})
    store.add("carrot celery", {"kind": "veg"})

    results = store.query("apple", top_k=5)
    assert len(results) == 1
    assert results[0].metadata["kind"] == "fruit"


@pytest.mark.asyncio
async def test_memory_tool_add_and_search() -> None:
    tool = MemoryTool()

    out_add = await tool(action="add", text="alpha beta", metadata={"id": 1})
    assert isinstance(out_add[0], TextContent)

    out_search = await tool(action="search", text="alpha")
    assert isinstance(out_search[0], TextContent)
    assert out_search[0].text.startswith("1.")


@pytest.mark.asyncio
async def test_memory_tool_unknown_action() -> None:
    tool = MemoryTool()
    res = await tool(action="noop", text="x")
    assert isinstance(res[0], TextContent)
    assert res[0].text == "unknown action"
