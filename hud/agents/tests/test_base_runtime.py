from __future__ import annotations

from typing import Any
from unittest import mock

import mcp.types as types
import pytest

from hud.agents.base import BaseCreateParams, MCPAgent, find_content, find_reward, text_to_blocks
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult

from .conftest import MockMCPClient


class DummyConfig(BaseAgentConfig):
    model_name: str = "DummyAgent"
    checkpoint_name: str = "dummy-model"


class DummyCreateParams(BaseCreateParams, DummyConfig):
    pass


class DummyAgent(MCPAgent):
    config_cls = DummyConfig

    def __init__(self, **kwargs: Any) -> None:
        # Only create MockMCPClient if mcp_client not specified at all
        if "mcp_client" not in kwargs:
            kwargs["mcp_client"] = MockMCPClient()
        params = DummyCreateParams(**kwargs)
        super().__init__(params)

    async def get_system_messages(self) -> list[types.ContentBlock]:
        return [types.TextContent(type="text", text="sys")]

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
    fake_client = MockMCPClient()

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
    call_count = [0]
    ok_result = MCPToolResult(content=text_to_blocks("ok"), isError=False)

    def handler(tool_call: MCPToolCall) -> MCPToolResult:
        call_count[0] += 1
        if call_count[0] == 1:
            return ok_result
        raise RuntimeError("boom")

    fake_client = MockMCPClient(call_tool_handler=handler)
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
    agent = DummyAgent(mcp_client=MockMCPClient(), auto_trace=False)
    with pytest.raises(RuntimeError):
        agent.get_available_tools()


@pytest.mark.asyncio
async def test_format_message_invalid_type_raises():
    agent = DummyAgent(mcp_client=MockMCPClient(), auto_trace=False)
    with pytest.raises(ValueError):
        await agent.format_message({"oops": 1})  # type: ignore


@pytest.mark.asyncio
async def test_call_tools_timeout_error_shutdown_called():
    def handler(tool_call: MCPToolCall) -> MCPToolResult:
        raise TimeoutError("timeout")

    fake_client = MockMCPClient(call_tool_handler=handler)
    agent = DummyAgent(mcp_client=fake_client, auto_trace=False)
    with pytest.raises(TimeoutError):
        await agent.call_tools(MCPToolCall(name="x", arguments={}))
    assert fake_client.shutdown_called


def test_text_to_blocks_shapes():
    blocks = text_to_blocks("x")
    assert isinstance(blocks, list) and blocks and isinstance(blocks[0], types.TextContent)


@pytest.mark.asyncio
async def test_run_returns_connection_error_trace(monkeypatch):
    fake_client = MockMCPClient(
        initialize_error=RuntimeError("Connection refused http://localhost:1234")
    )

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
    ok = MCPToolResult(content=text_to_blocks("ok"), isError=False)
    fake_client = MockMCPClient(call_tool_handler=lambda _: ok)

    class DummyCM:
        def __exit__(self, *args, **kwargs):
            return False

    monkeypatch.setattr("hud.utils.mcp.setup_hud_telemetry", lambda *args, **kwargs: DummyCM())

    agent = DummyAgent(mcp_client=fake_client, auto_trace=False, response_tool_name="submit")
    result = await agent.run("hello", max_steps=1)
    assert result.isError is False
    assert len(fake_client.call_tool_calls) > 0


@pytest.mark.asyncio
async def test_get_available_tools_after_initialize(monkeypatch):
    fake_client = MockMCPClient()

    class DummyCM:
        def __exit__(self, *args, **kwargs):
            return False

    monkeypatch.setattr("hud.utils.mcp.setup_hud_telemetry", lambda *args, **kwargs: DummyCM())

    agent = DummyAgent(mcp_client=fake_client, auto_trace=False)
    await agent.initialize(None)
    assert agent.get_available_tools() == []
