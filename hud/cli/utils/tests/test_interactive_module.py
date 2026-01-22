from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hud.cli.utils.interactive import InteractiveMCPTester


@pytest.mark.asyncio
@patch("fastmcp.Client")
async def test_connect_and_disconnect(MockClient):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.list_tools.return_value = []
    client.is_connected.return_value = True
    client.close = AsyncMock()
    MockClient.return_value = client

    tester = InteractiveMCPTester("http://localhost:8765/mcp", verbose=False)
    ok = await tester.connect()
    assert ok is True
    assert tester.tools == []
    await tester.disconnect()
    client.close.assert_called_once()


def test_display_tools_handles_empty(capfd):
    tester = InteractiveMCPTester("http://x")
    tester.tools = []
    tester.display_tools()  # prints warning


@pytest.mark.asyncio
@patch("hud.cli.utils.interactive.questionary")
async def test_select_tool_quit(mock_questionary):
    tester = InteractiveMCPTester("http://x")
    tester.tools = [SimpleNamespace(name="a", description="")]
    # Simulate ESC/quit
    mock_questionary.select.return_value.unsafe_ask_async.return_value = "❌ Quit"
    sel = await tester.select_tool()
    assert sel is None


@pytest.mark.asyncio
@patch("hud.cli.utils.interactive.console")
async def test_get_tool_arguments_no_schema(mock_console):
    tester = InteractiveMCPTester("http://x")
    args = await tester.get_tool_arguments(SimpleNamespace(name="t", inputSchema=None))
    assert args == {}


@pytest.mark.asyncio
@patch("hud.cli.utils.interactive.console")
async def test_call_tool_success(mock_console):
    tester = InteractiveMCPTester("http://x")
    fake_result = SimpleNamespace(is_error=False, content=[SimpleNamespace(text="ok")])
    tester.client = AsyncMock()
    tester.client.call_tool.return_value = fake_result
    await tester.call_tool(SimpleNamespace(name="t"), {"a": 1})
    assert tester.client.call_tool.awaited
