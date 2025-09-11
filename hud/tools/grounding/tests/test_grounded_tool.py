from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mcp.types as types
import pytest

from hud.tools.grounding.grounded_tool import GroundedComputerTool
from hud.types import MCPToolCall, MCPToolResult


@dataclass
class FakeResult:
    content: list[types.ContentBlock]
    isError: bool = False
    structuredContent: dict | None = None


class FakeMCPClient:
    """Fake MCP client that implements AgentMCPClient protocol."""

    _initialized: bool

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._initialized = False

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        return {"test": {"command": "echo", "args": ["test"]}}

    @property
    def is_connected(self) -> bool:
        return self._initialized

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        self._initialized = True

    async def list_tools(self) -> list[types.Tool]:
        return [types.Tool(name="computer", description="Test tool", inputSchema={})]

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        self.calls.append((tool_call.name, tool_call.arguments or {}))
        return MCPToolResult(content=[types.TextContent(text="ok", type="text")], isError=False)

    async def shutdown(self) -> None:
        self._initialized = False


class FakeGrounder:
    """Fake grounder that implements Grounder interface."""

    def __init__(self, coords: tuple[int, int] | None = (10, 20)) -> None:
        self.coords = coords
        self.calls: list[tuple[str, str]] = []

    async def predict_click(
        self, *, image_b64: str, instruction: str, max_retries: int = 3
    ) -> tuple[int, int] | None:
        self.calls.append((image_b64[:10], instruction))
        return self.coords


def _png_b64() -> str:
    # 1x1 transparent PNG base64 (valid minimal image)
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB"
        "J2n0mQAAAABJRU5ErkJggg=="
    )


@pytest.mark.asyncio
async def test_click_action_grounds_and_calls_mcp() -> None:
    client = FakeMCPClient()
    grounder = FakeGrounder(coords=(123, 456))
    tool = GroundedComputerTool(grounder=grounder, mcp_client=client)  # type: ignore

    blocks = await tool(
        action="click",
        element_description="red button",
        screenshot_b64=_png_b64(),
        button="left",
    )

    assert isinstance(blocks, list)
    # Grounder called once
    assert len(grounder.calls) == 1
    # MCP called with resolved coordinates
    assert client.calls == [("computer", {"action": "click", "x": 123, "y": 456, "button": "left"})]


@pytest.mark.asyncio
async def test_move_and_scroll_require_element_description_and_screenshot() -> None:
    client = FakeMCPClient()
    grounder = FakeGrounder(coords=(5, 6))
    tool = GroundedComputerTool(grounder=grounder, mcp_client=client)  # type: ignore

    # Missing element_description
    with pytest.raises(Exception) as ei:
        await tool(action="move", screenshot_b64=_png_b64())
    assert "element_description is required" in str(ei.value)

    # Missing screenshot
    with pytest.raises(Exception) as ei2:
        await tool(action="scroll", element_description="list", scroll_y=100)
    assert "No screenshot available" in str(ei2.value)


@pytest.mark.asyncio
async def test_drag_grounds_both_points_and_calls_mcp() -> None:
    client = FakeMCPClient()
    grounder = FakeGrounder(coords=(10, 20))
    tool = GroundedComputerTool(grounder=grounder, mcp_client=client)  # type: ignore

    await tool(
        action="drag",
        start_element_description="source",
        end_element_description="target",
        screenshot_b64=_png_b64(),
        button="left",
    )

    # Two grounding calls (start and end)
    assert len(grounder.calls) == 2
    # Drag path contains two points, same coords from fake grounder
    name, args = client.calls[0]
    assert name == "computer"
    assert args["action"] == "drag"
    assert args["button"] == "left"
    assert args["path"] == [(10, 20), (10, 20)]


@pytest.mark.asyncio
async def test_drag_requires_both_descriptions_and_screenshot() -> None:
    client = FakeMCPClient()
    grounder = FakeGrounder()
    tool = GroundedComputerTool(grounder=grounder, mcp_client=client)  # type: ignore

    with pytest.raises(Exception) as ei:
        await tool(action="drag", start_element_description="a", screenshot_b64=_png_b64())
    assert "start_element_description and end_element_description" in str(ei.value)

    with pytest.raises(Exception) as ei2:
        await tool(
            action="drag",
            start_element_description="a",
            end_element_description="b",
        )
    assert "No screenshot available" in str(ei2.value)


@pytest.mark.asyncio
async def test_direct_actions_bypass_grounding_and_call_mcp() -> None:
    client = FakeMCPClient()
    grounder = FakeGrounder()
    tool = GroundedComputerTool(grounder=grounder, mcp_client=client)  # type: ignore

    # Actions that bypass grounding
    for action, extra in [
        ("screenshot", {}),
        ("type", {"text": "hello"}),
        ("keypress", {"keys": ["ctrl", "a"]}),
        ("wait", {}),
        ("get_current_url", {}),
        ("get_dimensions", {}),
        ("get_environment", {}),
    ]:
        client.calls.clear()
        _ = await tool(action=action, **extra)
        assert client.calls and client.calls[0][0] == "computer"
        assert client.calls[0][1]["action"] == action
    # Grounder not invoked for these
    assert grounder.calls == []


@pytest.mark.asyncio
async def test_unsupported_action_raises() -> None:
    client = FakeMCPClient()
    grounder = FakeGrounder()
    tool = GroundedComputerTool(grounder=grounder, mcp_client=client)  # type: ignore

    with pytest.raises(Exception) as ei:
        await tool(action="zoom")
    assert "Unsupported action" in str(ei.value)


@pytest.mark.asyncio
async def test_grounding_failure_propagates_as_error() -> None:
    client = FakeMCPClient()
    grounder = FakeGrounder(coords=None)
    tool = GroundedComputerTool(grounder=grounder, mcp_client=client)  # type: ignore

    with pytest.raises(Exception) as ei:
        await tool(action="click", element_description="x", screenshot_b64=_png_b64())
    assert "Could not locate element" in str(ei.value)
