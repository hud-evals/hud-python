from __future__ import annotations

from typing import Any

import mcp.types as types
import pytest
from openai import AsyncOpenAI

from hud.agents.grounded_openai import GroundedOpenAIChatAgent
from hud.tools.grounding import GrounderConfig
from hud.types import MCPToolCall, MCPToolResult


class FakeMCPClient:
    def __init__(self) -> None:
        self.tools: list[types.Tool] = [
            types.Tool(name="computer", description="", inputSchema={}),
            types.Tool(name="setup", description="internal functions", inputSchema={}),
        ]
        self.called: list[MCPToolCall] = []
        self._initialized = True

    async def initialize(self, mcp_config: dict[str, dict[str, Any]] | None = None) -> None:
        return None

    async def list_tools(self) -> list[types.Tool]:
        return self.tools

    async def call_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        self.called.append(tool_call)
        return MCPToolResult(content=[types.TextContent(text="ok", type="text")], isError=False)

    @property
    def mcp_config(self) -> dict[str, dict[str, Any]]:
        return {"local": {"command": "echo", "args": ["ok"]}}

    @property
    def is_connected(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        return None

    async def list_resources(self) -> list[types.Resource]:  # not used here
        return []

    async def read_resource(self, uri: str) -> types.ReadResourceResult | None:
        return None


class DummyGrounder:
    async def predict_click(self, *, image_b64: str, instruction: str, max_retries: int = 3):
        return (7, 9)


class DummyGroundedTool:
    def __init__(self) -> None:
        self.last_args: dict[str, Any] | None = None

    async def __call__(self, **kwargs: Any):
        self.last_args = kwargs
        return [types.TextContent(text="ok", type="text")]

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": "computer", "parameters": {"type": "object"}},
        }


@pytest.mark.asyncio
async def test_call_tools_injects_screenshot_and_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    # Agent with fake OpenAI client and fake MCP client
    grounder_cfg = GrounderConfig(api_base="http://example", model="qwen")
    fake_openai = AsyncOpenAI(api_key="test")
    agent = GroundedOpenAIChatAgent.create(
        grounder_config=grounder_cfg,
        openai_client=fake_openai,
        checkpoint_name="gpt-4o-mini",
        mcp_client=FakeMCPClient(),
        initial_screenshot=False,
    )

    # Inject a dummy grounded tool to observe args without full initialization
    dummy_tool = DummyGroundedTool()
    agent.grounded_tool = dummy_tool  # type: ignore

    # Seed conversation history with a user image
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB"
        "J2n0mQAAAABJRU5ErkJggg=="
    )
    agent.conversation_history = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
            ],
        }
    ]

    # Build a tool call as GroundedOpenAIChatAgent.get_response would produce
    tool_call = MCPToolCall(
        name="computer", arguments={"action": "click", "element_description": "blue button"}
    )

    results = await agent.call_tools(tool_call)

    # One result returned
    assert len(results) == 1 and not results[0].isError

    # Grounded tool received screenshot_b64 injected
    assert dummy_tool.last_args is not None
    assert dummy_tool.last_args["action"] == "click"
    assert dummy_tool.last_args["element_description"] == "blue button"
    assert "screenshot_b64" in dummy_tool.last_args
    assert isinstance(dummy_tool.last_args["screenshot_b64"], str)
