"""Extended tests for agent.py to improve coverage."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import types
from mcp.types import ImageContent, TextContent

from hud.agent import MCPAgent
from hud.datasets import TaskConfig
from hud.types import AgentResponse, MCPToolCall


class MockAgent(MCPAgent):
    """Mock agent for testing."""

    def __init__(self, **kwargs: Any):
        mcp_client = kwargs.pop("mcp_client", None)
        if mcp_client is None:
            mcp_client = MagicMock()
            mcp_client.initialize = AsyncMock()
            mcp_client.list_tools = AsyncMock(return_value=[])
            mcp_client.call_tool = AsyncMock()
        super().__init__(mcp_client=mcp_client, **kwargs)

    async def run(self, task: TaskConfig) -> Any:
        return {"result": "mock"}

    def create_initial_messages(
        self, prompt: str, screenshot: str | None = None
    ) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt}]

    def get_model_response(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"role": "assistant", "content": "Mock response"}

    def format_tool_results(
        self,
        results: list[tuple[str, Any]],
        screenshot: str | None = None,
        assistant_msg: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [{"role": "tool", "content": str(results)}]


class TestMCPAgentExtended:
    """Extended tests for MCPAgent functionality."""

    @pytest.mark.asyncio
    async def test_initialize_with_task_setup_tools(self):
        """Test initialization with TaskConfig containing setup tools."""

        # Create task with setup tools
        setup_call = MCPToolCall(name="setup", arguments={"mode": "test"})
        task = TaskConfig(prompt="Test task", mcp_config={"test": "config"}, setup_tool=setup_call)

        agent = MockAgent()

        # Mock list_tools to return the setup tool
        agent.mcp_client.list_tools.return_value = [
            types.Tool(name="setup", description="Setup", inputSchema={"type": "object"})
        ]

        await agent.initialize(task)

        # Verify setup tool added to lifecycle tools
        assert "setup" in agent.lifecycle_tools

        # Verify it's excluded from available tools
        available = agent.get_available_tools()
        assert not any(t.name == "setup" for t in available)

    @pytest.mark.asyncio
    async def test_initialize_with_multiple_setup_tools(self):
        """Test initialization with multiple setup tools."""

        setup_calls = [
            MCPToolCall(name="setup1", arguments={}),
            MCPToolCall(name="setup2", arguments={}),
        ]

        task = TaskConfig(prompt="Test task", mcp_config={"test": "config"}, setup_tool=setup_calls)

        agent = MockAgent()
        await agent.initialize(task)

        assert "setup1" in agent.lifecycle_tools
        assert "setup2" in agent.lifecycle_tools

    @pytest.mark.asyncio
    async def test_initialize_with_evaluate_tools(self):
        """Test initialization with evaluate tools."""

        eval_calls = [
            MCPToolCall(name="eval1", arguments={}),
            MCPToolCall(name="eval2", arguments={}),
        ]

        task = TaskConfig(
            prompt="Test task", mcp_config={"test": "config"}, evaluate_tool=eval_calls
        )

        agent = MockAgent()
        await agent.initialize(task)

        assert "eval1" in agent.lifecycle_tools
        assert "eval2" in agent.lifecycle_tools

    def test_has_computer_tools(self):
        """Test has_computer_tools method."""

        agent = MockAgent()

        # No tools - should return False
        agent._tool_map = {}
        agent._available_tools = []
        assert not agent.has_computer_tools()

        # Non-computer tools
        other_tool = MagicMock()
        other_tool.name = "other_tool"
        agent._tool_map = {"other_tool": other_tool}
        agent._available_tools = [other_tool]
        assert not agent.has_computer_tools()

        # With computer tool
        computer_tool = MagicMock()
        computer_tool.name = "computer"
        agent._tool_map = {"computer": computer_tool}
        agent._available_tools = [computer_tool]
        assert agent.has_computer_tools()

        # With screenshot tool
        screenshot_tool = MagicMock()
        screenshot_tool.name = "screenshot"
        agent._tool_map = {"screenshot": screenshot_tool}
        agent._available_tools = [screenshot_tool]
        assert agent.has_computer_tools()

        # With various computer tool variants
        # Only test tool names that are actually in the computer_tools set in has_computer_tools()
        for tool_name in ["computer_anthropic", "computer_openai"]:
            tool = MagicMock()
            tool.name = tool_name
            agent._tool_map = {tool_name: tool}
            agent._available_tools = [tool]
            assert agent.has_computer_tools()

    def test_call_tool_validation(self):
        """Test call_tool parameter validation."""

        agent = MockAgent()

        # Test empty tool name
        with pytest.raises(ValueError, match="Tool call must have a 'name' field"):
            import asyncio

            asyncio.run(agent.call_tool(MCPToolCall(name="", arguments={})))

    @pytest.mark.asyncio
    async def test_call_tool_not_allowed(self):
        """Test calling a tool that's not allowed."""

        agent = MockAgent(allowed_tools=["allowed_tool"])
        # Only put allowed tool in _tool_map, forbidden tool should not be there after filtering
        agent._tool_map = {"allowed_tool": MagicMock()}

        with pytest.raises(ValueError, match="Tool 'forbidden_tool' not found or not allowed"):
            await agent.call_tool(MCPToolCall(name="forbidden_tool", arguments={}))

    @pytest.mark.asyncio
    async def test_capture_screenshot_no_tools(self):
        """Test capture_screenshot when no computer tools available."""

        agent = MockAgent()
        agent._tool_map = {}

        result = await agent.capture_screenshot()
        assert result is None

    @pytest.mark.asyncio
    async def test_capture_screenshot_with_error(self):
        """Test capture_screenshot handling errors."""

        agent = MockAgent()
        agent._tool_map = {"screenshot": MagicMock()}

        # Mock tool call to raise exception
        agent.mcp_client.call_tool.side_effect = Exception("Screenshot failed")

        result = await agent.capture_screenshot()
        assert result is None

    @pytest.mark.asyncio
    async def test_capture_screenshot_with_text_result(self):
        """Test capture_screenshot with text content result."""

        agent = MockAgent()
        agent._tool_map = {"screenshot": MagicMock()}

        # Mock tool returns text instead of image
        agent.mcp_client.call_tool.return_value = types.CallToolResult(
            content=[TextContent(type="text", text="Not an image")], isError=False
        )

        result = await agent.capture_screenshot()
        assert result is None

    @pytest.mark.asyncio
    async def test_capture_screenshot_computer_openai(self):
        """Test capture_screenshot with OpenAI computer tool."""

        agent = MockAgent()
        # Create the tool and add to both _tool_map and _available_tools
        openai_tool = MagicMock()
        openai_tool.name = "computer_openai"
        agent._tool_map = {"computer_openai": openai_tool}
        agent._available_tools = [openai_tool]

        # Mock successful screenshot
        agent.mcp_client.call_tool.return_value = types.CallToolResult(
            content=[ImageContent(type="image", data="base64data", mimeType="image/png")],
            isError=False,
        )

        result = await agent.capture_screenshot()
        assert result == "base64data"

        # Verify correct arguments for OpenAI
        agent.mcp_client.call_tool.assert_called_once()
        call_args = agent.mcp_client.call_tool.call_args
        assert call_args[0] == ("computer_openai", {"type": "screenshot"})

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""

        # Try to instantiate MCPAgent directly
        with pytest.raises(TypeError):
            MCPAgent(mcp_client=MagicMock())

    @pytest.mark.asyncio
    async def test_full_agent_response_flow(self):
        """Test full agent response flow with tool calls."""

        # Create a more complete mock agent
        class TestAgent(MCPAgent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.messages = []

            async def run(self, task: TaskConfig) -> AgentResponse:
                messages = self.create_initial_messages(task.prompt)

                # Simulate getting response with tool call
                response = self.get_model_response(messages)

                # Simulate tool execution
                tool_results = [("test_tool", {"result": "success"})]

                # Format results
                formatted = self.format_tool_results(tool_results, assistant_msg=response)

                return AgentResponse(
                    done=True,
                    content="I'll help you.",
                    info={"test": True, "messages": messages + [response] + formatted},
                )

            def create_initial_messages(
                self, prompt: str, screenshot: str | None = None
            ) -> list[dict[str, Any]]:
                msgs = [{"role": "user", "content": prompt}]
                if screenshot:
                    msgs.append({"role": "system", "content": f"Screenshot: {screenshot}"})
                return msgs

            def get_model_response(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
                return {
                    "role": "assistant",
                    "content": "I'll help you.",
                    "tool_calls": [{"name": "test_tool"}],
                }

            def format_tool_results(
                self,
                results: list[tuple[str, Any]],
                screenshot: str | None = None,
                assistant_msg: dict[str, Any] | None = None,
            ) -> list[dict[str, Any]]:
                formatted = []
                for tool_name, result in results:
                    formatted.append({"role": "tool", "name": tool_name, "content": str(result)})
                if screenshot:
                    formatted.append({"role": "system", "content": f"Screenshot: {screenshot}"})
                return formatted

        # Initialize client
        client = MagicMock()
        client.initialize = AsyncMock()
        client.list_tools = AsyncMock(return_value=[])

        agent = TestAgent(mcp_client=client)
        await agent.initialize()

        # Run task
        task = TaskConfig(prompt="Help me with this", mcp_config={})
        response = await agent.run(task)

        assert isinstance(response, AgentResponse)
        assert response.done is True
        assert response.content == "I'll help you."
        assert response.info["test"] is True
        assert len(response.info["messages"]) > 0
