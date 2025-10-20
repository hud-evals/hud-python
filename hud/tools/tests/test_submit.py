from __future__ import annotations

import pytest
from mcp.types import TextContent

from hud.tools.submit import SubmitTool, get_submission, set_submission


@pytest.fixture(autouse=True)
def reset_submission():
    """Reset submission before each test."""
    set_submission(None)
    yield
    set_submission(None)


def test_set_and_get_submission():
    """Test setting and getting submission value."""
    assert get_submission() is None

    set_submission("test value")
    assert get_submission() == "test value"

    set_submission("another value")
    assert get_submission() == "another value"

    set_submission(None)
    assert get_submission() is None


@pytest.mark.asyncio
async def test_submit_tool_with_response():
    """Test SubmitTool with a response string."""
    tool = SubmitTool()

    result = await tool(response="Test response")

    assert get_submission() == "Test response"
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "Test response"


@pytest.mark.asyncio
async def test_submit_tool_with_none():
    """Test SubmitTool with None response."""
    tool = SubmitTool()

    result = await tool(response=None)

    assert get_submission() is None
    assert len(result) == 0


@pytest.mark.asyncio
async def test_submit_tool_with_empty_string():
    """Test SubmitTool with empty string."""
    tool = SubmitTool()

    result = await tool(response="")

    assert get_submission() == ""
    assert len(result) == 0


@pytest.mark.asyncio
async def test_submit_tool_overwrite():
    """Test that submitting overwrites previous submission."""
    tool = SubmitTool()

    await tool(response="First submission")
    assert get_submission() == "First submission"

    await tool(response="Second submission")
    assert get_submission() == "Second submission"


@pytest.mark.asyncio
async def test_submit_tool_properties():
    """Test SubmitTool properties."""
    tool = SubmitTool()

    assert tool.name == "response"
    assert tool.title == "Submit Tool"
    assert "final response" in tool.description.lower()
