from __future__ import annotations

from unittest import mock

import mcp.types as types
import pytest

from hud.agents.base import MCPAgent, find_content, find_reward, text_to_blocks
from hud.types import AgentResponse, MCPToolCall, MCPToolResult


class DummyAgent(MCPAgent):
    async def get_system_messages(self):
        return [types.TextContent(text="sys", type="text")]

    async def get_response(self, messages):
        # Single step: no tool calls -> done
        return AgentResponse(content="ok", tool_calls=[], done=True)

    async def format_blocks(self, blocks):
        # Return as-is
        return blocks

    async def format_tool_results(self, tool_calls, tool_results):
        return [types.TextContent(text="tools", type="text")]


@pytest.mark.asyncio
async def test_run_with_string_prompt_auto_client(monkeypatch):
    # Fake MCPClient with required methods
    fake_client = mock.AsyncMock()
    fake_client.initialize.return_value = None
    fake_client.list_tools.return_value = []
    fake_client.shutdown.return_value = None

    # Patch MCPClient construction inside initialize()
    with mock.patch("hud.clients.MCPClient", return_value=fake_client):
        agent = DummyAgent(mcp_client=fake_client, auto_trace=False)
        result = await agent.run("hello", max_steps=1)
    assert result.done is True and result.isError is False


def test_find_reward_and_content_extractors():
    # Structured content
    r = MCPToolResult(
        content=text_to_blocks("{}"), isError=False, structuredContent={"reward": 0.7}
    )
    assert find_reward(r) == 0.7

    # Text JSON
    r2 = MCPToolResult(content=text_to_blocks('{"score": 0.5, "content": "hi"}'), isError=False)
    assert find_reward(r2) == 0.5
    assert find_content(r2) == "hi"


@pytest.mark.asyncio
async def test_call_tools_error_paths():
    fake_client = mock.AsyncMock()
    # First call succeeds
    ok_result = MCPToolResult(content=text_to_blocks("ok"), isError=False)
    fake_client.call_tool.side_effect = [ok_result, RuntimeError("boom")]
    agent = DummyAgent(mcp_client=fake_client, auto_trace=False)
    results = await agent.call_tools(
        [MCPToolCall(name="a", arguments={}), MCPToolCall(name="b", arguments={})]
    )
    assert results[0].isError is False
    assert results[1].isError is True


@pytest.mark.asyncio
async def test_initialize_without_client_raises_valueerror():
    agent = DummyAgent(mcp_client=None, auto_trace=False)
    with pytest.raises(ValueError):
        await agent.initialize(None)


def test_get_available_tools_before_initialize_raises():
    agent = DummyAgent(mcp_client=mock.AsyncMock(), auto_trace=False)
    with pytest.raises(RuntimeError):
        agent.get_available_tools()


@pytest.mark.asyncio
async def test_format_message_invalid_type_raises():
    agent = DummyAgent(mcp_client=mock.AsyncMock(), auto_trace=False)
    with pytest.raises(ValueError):
        await agent.format_message({"oops": 1})  # type: ignore


@pytest.mark.asyncio
async def test_call_tools_timeout_error_shutdown_called():
    fake_client = mock.AsyncMock()
    fake_client.call_tool.side_effect = TimeoutError("timeout")
    fake_client.shutdown.return_value = None
    agent = DummyAgent(mcp_client=fake_client, auto_trace=False)
    with pytest.raises(TimeoutError):
        await agent.call_tools(MCPToolCall(name="x", arguments={}))
    fake_client.shutdown.assert_awaited_once()


def test_text_to_blocks_shapes():
    blocks = text_to_blocks("x")
    assert isinstance(blocks, list) and blocks and isinstance(blocks[0], types.TextContent)


@pytest.mark.asyncio
async def test_run_returns_connection_error_trace(monkeypatch):
    fake_client = mock.AsyncMock()
    fake_client.mcp_config = {}
    fake_client.initialize.side_effect = RuntimeError("Connection refused http://localhost:1234")
    fake_client.list_tools.return_value = []
    fake_client.shutdown.return_value = None

    class DummyCM:
        def __exit__(self, *args, **kwargs):
            return False

    monkeypatch.setattr("hud.utils.mcp.setup_hud_telemetry", lambda *args, **kwargs: DummyCM())

    agent = DummyAgent(mcp_client=fake_client, auto_trace=False)
    result = await agent.run("p", max_steps=1)
    assert result.isError is True
    assert "Could not connect" in (result.content or "")


@pytest.mark.asyncio
async def test_run_calls_response_tool_when_configured(monkeypatch):
    fake_client = mock.AsyncMock()
    fake_client.mcp_config = {}
    fake_client.initialize.return_value = None
    fake_client.list_tools.return_value = []
    fake_client.shutdown.return_value = None
    ok = MCPToolResult(content=text_to_blocks("ok"), isError=False)
    fake_client.call_tool.return_value = ok

    class DummyCM:
        def __exit__(self, *args, **kwargs):
            return False

    monkeypatch.setattr("hud.utils.mcp.setup_hud_telemetry", lambda *args, **kwargs: DummyCM())

    agent = DummyAgent(mcp_client=fake_client, auto_trace=False, response_tool_name="submit")
    result = await agent.run("hello", max_steps=1)
    assert result.isError is False
    fake_client.call_tool.assert_awaited()


@pytest.mark.asyncio
async def test_get_available_tools_after_initialize(monkeypatch):
    fake_client = mock.AsyncMock()
    fake_client.mcp_config = {}
    fake_client.initialize.return_value = None
    fake_client.list_tools.return_value = []
    fake_client.shutdown.return_value = None

    class DummyCM:
        def __exit__(self, *args, **kwargs):
            return False

    monkeypatch.setattr("hud.utils.mcp.setup_hud_telemetry", lambda *args, **kwargs: DummyCM())

    agent = DummyAgent(mcp_client=fake_client, auto_trace=False)
    await agent.initialize(None)
    assert agent.get_available_tools() == []
