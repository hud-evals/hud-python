"""Tests for Gemini memory tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from mcp.types import TextContent

from hud.tools.memory import GeminiMemoryTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def memory_tool(tmp_path: Path) -> GeminiMemoryTool:
    """Create a GeminiMemoryTool with a temporary directory."""
    return GeminiMemoryTool(memory_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_gemini_memory_save_fact(memory_tool: GeminiMemoryTool) -> None:
    """Test saving a fact to memory."""
    result = await memory_tool(fact="User prefers dark mode")

    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "remembered" in result[0].text.lower()
    assert "User prefers dark mode" in result[0].text


@pytest.mark.asyncio
async def test_gemini_memory_creates_section(memory_tool: GeminiMemoryTool) -> None:
    """Test that saving creates the memory section header."""
    await memory_tool(fact="First fact")

    content = memory_tool._memory_file.read_text()
    assert "## Gemini Added Memories" in content
    assert "- First fact" in content


@pytest.mark.asyncio
async def test_gemini_memory_appends_facts(memory_tool: GeminiMemoryTool) -> None:
    """Test that multiple facts are appended correctly."""
    await memory_tool(fact="Fact one")
    await memory_tool(fact="Fact two")
    await memory_tool(fact="Fact three")

    content = memory_tool._memory_file.read_text()
    assert "- Fact one" in content
    assert "- Fact two" in content
    assert "- Fact three" in content


@pytest.mark.asyncio
async def test_gemini_memory_get_all_memories(memory_tool: GeminiMemoryTool) -> None:
    """Test retrieving all memories."""
    await memory_tool(fact="Remember this")
    await memory_tool(fact="And this too")

    memories = memory_tool.get_all_memories()
    assert len(memories) == 2
    assert "Remember this" in memories
    assert "And this too" in memories


@pytest.mark.asyncio
async def test_gemini_memory_empty_fact_error(memory_tool: GeminiMemoryTool) -> None:
    """Test that empty fact raises error."""
    from hud.tools.types import ToolError

    with pytest.raises(ToolError, match="non-empty"):
        await memory_tool(fact="")


@pytest.mark.asyncio
async def test_gemini_memory_strips_leading_dashes(memory_tool: GeminiMemoryTool) -> None:
    """Test that leading dashes are stripped from facts."""
    await memory_tool(fact="- Already has dash")

    content = memory_tool._memory_file.read_text()
    # Should not have double dash
    assert "- - Already" not in content
    assert "- Already has dash" in content
