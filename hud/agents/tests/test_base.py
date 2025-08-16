"""Tests for BaseMCPAgent using simulated actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

# Import AsyncMock from unittest.mock if available (Python 3.8+)
try:
    from unittest.mock import AsyncMock
except ImportError:
    # Fallback for older Python versions
    from unittest.mock import MagicMock as AsyncMock

import pytest
from mcp import types

from hud.agent import MCPAgent
from hud.tools.executors.base import BaseExecutor
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace

if TYPE_CHECKING:
    from hud.datasets import TaskConfig as Task

from hud.datasets import TaskConfig


class MockMCPAgent(MCPAgent):
    """Concrete implementation of BaseMCPAgent for testing."""

    def __init__(self, mcp_client: Any = None, **kwargs: Any) -> None:
        if mcp_client is None:
            # Create a mock client if none provided
            mcp_client = MagicMock()
            mcp_client.get_available_tools = MagicMock(return_value=[])
            mcp_client.initialize = AsyncMock()
            mcp_client.list_tools = AsyncMock(return_value=[])
        super().__init__(mcp_client=mcp_client, **kwargs)
        self.executor = BaseExecutor()  # Use simulated executor
        self._messages = []

    async def run(self, task: Task) -> list[dict[str, Any]]:
        """Mock run method."""
        return self._messages

    async def create_initial_messages(
        self, prompt: str, initial_screenshot: bool = False
    ) -> list[dict[str, Any]]:
        """Mock create initial messages."""
        messages = [{"role": "user", "content": prompt}]
        if initial_screenshot:
            messages.append({"role": "assistant", "content": "Screenshot: mock_screenshot"})
        return messages

    async def get_model_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        """Mock get model response."""
        return AgentResponse(content="Mock response", tool_calls=[], done=True)

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        """Mock format tool results."""
        formatted = []
        for tool_call, result in zip(tool_calls, tool_results):
            formatted.append({"role": "tool", "name": tool_call.name, "content": str(result)})
        return formatted

    async def create_user_message(self, text: str) -> Any:
        """Mock create user message."""
        return {"role": "user", "content": text}


class TestBaseMCPAgent:
    """Tests for BaseMCPAgent with simulated actions."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        agent = MockMCPAgent()

        assert agent.mcp_client is not None
        assert agent.allowed_tools is None
        assert agent.disallowed_tools == []
        assert agent.initial_screenshot is False
        assert agent.max_screenshot_history == 3
        assert agent.append_tool_system_prompt is True
        assert agent.custom_system_prompt is None
        assert agent.lifecycle_tools == []

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        client = MagicMock()
        agent = MockMCPAgent(
            mcp_client=client,
            allowed_tools=["tool1", "tool2"],
            disallowed_tools=["bad_tool"],
            initial_screenshot=True,
            max_screenshot_history=5,
            append_tool_system_prompt=False,
            custom_system_prompt="Custom prompt",
            lifecycle_tools=["custom_setup", "custom_eval"],
        )

        assert agent.mcp_client == client
        assert agent.allowed_tools == ["tool1", "tool2"]
        assert agent.disallowed_tools == ["bad_tool"]
        assert agent.initial_screenshot is True
        assert agent.max_screenshot_history == 5
        assert agent.append_tool_system_prompt is False
        assert agent.custom_system_prompt == "Custom prompt"
        assert agent.lifecycle_tools == ["custom_setup", "custom_eval"]

    def test_init_no_client(self):
        """Test init fails without client."""

        # Create a minimal concrete implementation to test the ValueError
        class TestAgent(MCPAgent):
            def create_initial_messages(
                self, prompt: str, screenshot: str | None = None
            ) -> list[dict[str, Any]]:
                return []

            def format_tool_results(
                self, results: list[tuple[str, Any]], screenshot: str | None = None
            ) -> list[dict[str, Any]]:
                return []

            async def get_model_response(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
                return {"content": "test"}

        with pytest.raises(ValueError, match="MCPClient is required"):
            TestAgent(mcp_client=None)

    @pytest.mark.asyncio
    async def test_initialize_with_sessions(self):
        """Test initialize with existing sessions."""
        agent = MockMCPAgent()

        # Create proper async mock for session
        mock_session = MagicMock()

        # Set up the connector and client_session structure
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        # Mock list_tools on the client_session
        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[
                    types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                    types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
                    types.Tool(
                        name="setup", description="Setup tool", inputSchema={"type": "object"}
                    ),
                ]
            )

        mock_session.connector.client_session.list_tools = mock_list_tools

        # Mock the list_tools method on mcp_client to return the tools
        agent.mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
                types.Tool(name="setup", description="Setup tool", inputSchema={"type": "object"}),
            ]
        )

        assert agent.mcp_client is not None

        await agent.initialize()

        # Check available tools were populated (excludes lifecycle tools)
        tools = agent.get_available_tools()
        assert len(tools) == 3  # All tools (setup is not in default lifecycle tools)

        # Ensure names exist in available tools
        names = {t.name for t in tools}
        assert {"tool1", "tool2", "setup"} <= names

    @pytest.mark.asyncio
    async def test_initialize_with_filtering(self):
        """Test initialize with tool filtering."""
        agent = MockMCPAgent(allowed_tools=["tool1"], disallowed_tools=["tool3"])

        # Create proper async mock for session
        mock_session = MagicMock()

        # Set up the connector and client_session structure
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[
                    types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                    types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
                    types.Tool(name="tool3", description="Tool 3", inputSchema={"type": "object"}),
                    types.Tool(name="setup", description="Setup", inputSchema={"type": "object"}),
                ]
            )

        mock_session.connector.client_session.list_tools = mock_list_tools

        assert agent.mcp_client is not None

        # Mock the list_tools method on mcp_client to return the tools
        agent.mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
                types.Tool(name="tool3", description="Tool 3", inputSchema={"type": "object"}),
                types.Tool(name="setup", description="Setup", inputSchema={"type": "object"}),
            ]
        )

        await agent.initialize()

        # Check filtering worked - get_available_tools excludes lifecycle tools
        tools = agent.get_available_tools()
        tool_names = [t.name for t in tools]
        assert len(tools) == 1  # Only tool1 (tool2 and tool3 are filtered out)
        assert "tool1" in tool_names
        assert "setup" not in tool_names  # Lifecycle tool excluded from available tools
        assert "tool2" not in tool_names  # Not in allowed list
        assert "tool3" not in tool_names  # In disallowed list

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        agent = MockMCPAgent()

        # Initialize with a tool
        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[
                    types.Tool(name="test_tool", description="Test", inputSchema={"type": "object"})
                ]
            )

        mock_session.connector.client_session.list_tools = mock_list_tools

        # Mock the call_tool method on the client session
        mock_result = types.CallToolResult(
            content=[types.TextContent(type="text", text="Tool result")], isError=False
        )

        async def mock_call_tool(name, args):
            return mock_result

        mock_session.connector.client_session.call_tool = mock_call_tool

        assert agent.mcp_client is not None

        # Mock the client's call_tool method directly
        agent.mcp_client.call_tool = AsyncMock(return_value=mock_result)

        # Mock the list_tools method to return the test tool
        agent.mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(name="test_tool", description="Test", inputSchema={"type": "object"})
            ]
        )

        await agent.initialize()

        # Call the tool
        tool_call = MCPToolCall(name="test_tool", arguments={"param": "value"})
        result = await agent.call_tool(tool_call)

        assert result == mock_result
        assert not result.isError

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        """Test calling non-existent tool."""
        agent = MockMCPAgent()

        # Initialize without tools
        mock_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(tools=[])

        mock_session.list_tools = mock_list_tools
        assert agent.mcp_client is not None

        await agent.initialize()

        # Try to call unknown tool
        with pytest.raises(ValueError, match="Tool 'unknown_tool' not found"):
            tool_call = MCPToolCall(name="unknown_tool", arguments={})
            await agent.call_tool(tool_call)

    @pytest.mark.asyncio
    async def test_call_tool_no_name(self):
        """Test calling tool without name."""
        # MCPToolCall accepts empty names, but the agent should validate
        agent = MockMCPAgent()
        tool_call = MCPToolCall(name="", arguments={})

        with pytest.raises(ValueError, match="Tool call must have a 'name' field"):
            await agent.call_tool(tool_call)

    def test_get_system_prompt_default(self):
        """Test get_system_prompt with default settings."""
        agent = MockMCPAgent()

        # Add some tools
        agent._available_tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
            types.Tool(name="setup", description="Setup", inputSchema={"type": "object"}),
        ]

        prompt = agent.get_system_prompt()

        # Should include ALL tool descriptions (including lifecycle tools)
        assert "tool1" in prompt
        assert "Tool 1" in prompt
        assert "setup" in prompt
        assert "Setup" in prompt

    def test_get_system_prompt_custom(self):
        """Test get_system_prompt with custom prompt."""
        agent = MockMCPAgent(
            custom_system_prompt="My custom prompt", append_tool_system_prompt=False
        )

        prompt = agent.get_system_prompt()
        assert prompt == "My custom prompt"

    def test_has_computer_tools(self):
        """Test checking for computer tools."""
        agent = MockMCPAgent()

        # No tools
        assert not agent.has_computer_tools()

        # With computer tool
        agent._available_tools = [
            types.Tool(name="computer", description="Computer", inputSchema={"type": "object"})
        ]
        assert agent.has_computer_tools()

        # With screenshot tool
        agent._available_tools = [
            types.Tool(name="screenshot", description="Screenshot", inputSchema={"type": "object"})
        ]
        assert agent.has_computer_tools()

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        agent = MockMCPAgent()

        # Add setup to lifecycle tools to test filtering
        agent.lifecycle_tools = ["setup"]

        agent._available_tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
            types.Tool(name="setup", description="Setup", inputSchema={"type": "object"}),
        ]

        schemas = agent.get_tool_schemas()

        # Should include non-lifecycle tools
        assert len(schemas) == 1
        assert schemas[0]["name"] == "tool1"

    @pytest.mark.asyncio
    async def test_capture_screenshot_no_tool(self):
        """Test screenshot capture without screenshot tool."""
        agent = MockMCPAgent()

        screenshot = await agent.capture_screenshot()
        assert screenshot is None

    @pytest.mark.asyncio
    async def test_capture_screenshot_with_tool(self):
        """Test screenshot capture with screenshot tool."""
        agent = MockMCPAgent()

        # Set up screenshot tool
        mock_session = MagicMock()
        mock_session.connector = MagicMock()
        mock_session.connector.client_session = MagicMock()

        async def mock_list_tools():
            return types.ListToolsResult(
                tools=[
                    types.Tool(
                        name="screenshot", description="Screenshot", inputSchema={"type": "object"}
                    )
                ]
            )

        mock_session.connector.client_session.list_tools = mock_list_tools

        # Mock screenshot result
        mock_result = types.CallToolResult(
            content=[
                types.ImageContent(type="image", data="base64imagedata", mimeType="image/png")
            ],
            isError=False,
        )

        async def mock_call_tool(name, args):
            return mock_result

        mock_session.connector.client_session.call_tool = mock_call_tool

        assert agent.mcp_client is not None

        # No tool map mocking required

        # Mock the client's call_tool method directly
        agent.mcp_client.call_tool = AsyncMock(return_value=mock_result)

        # Mock the list_tools method to return the screenshot tool
        agent.mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(
                    name="screenshot", description="Screenshot", inputSchema={"type": "object"}
                )
            ]
        )

        await agent.initialize()

        screenshot = await agent.capture_screenshot()
        assert screenshot == "base64imagedata"

    # process_tool_results method was removed from base class
    # This functionality is now handled internally

    def test_get_tools_by_server(self):
        """Test getting tools grouped by server."""
        agent = MockMCPAgent()

        # Set up tools from different servers
        tool1 = types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"})
        tool2 = types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"})

        agent._available_tools = [tool1, tool2]
        tools = agent.get_available_tools()
        assert {t.name for t in tools} == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_executor_integration(self):
        """Test integration with BaseExecutor for simulated actions."""
        agent = MockMCPAgent()

        # Test various executor actions
        click_result = await agent.executor.click(100, 200, take_screenshot=False)
        assert click_result.output is not None
        assert "[SIMULATED] Click at (100, 200)" in click_result.output

        type_result = await agent.executor.write("Test input", take_screenshot=False)
        assert type_result.output is not None
        assert "[SIMULATED] Type 'Test input'" in type_result.output

        scroll_result = await agent.executor.scroll(x=50, y=50, scroll_y=5, take_screenshot=False)
        assert scroll_result.output is not None
        assert "[SIMULATED] Scroll" in scroll_result.output

        # Test screenshot
        screenshot = await agent.executor.screenshot()
        assert isinstance(screenshot, str)
        assert screenshot.startswith("iVBORw0KGgo")  # PNG header


class MockAgentExtended(MCPAgent):
    """Mock agent for testing with predefined responses."""

    def __init__(self, responses=None, **kwargs):
        super().__init__(**kwargs)
        self.responses = responses or []
        self.call_count = 0

    async def create_initial_messages(
        self, prompt: str, initial_screenshot: bool = False
    ) -> list[dict[str, Any]]:
        """Create initial messages."""
        messages = [{"role": "user", "content": prompt}]
        if initial_screenshot:
            screenshot = await self.capture_screenshot()
            if screenshot:
                messages.append({"role": "assistant", "content": f"Screenshot: {screenshot}"})
        return messages

    async def get_model_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        """Return predefined responses - must be async."""
        if self.call_count < len(self.responses):
            response_dict = self.responses[self.call_count]
            self.call_count += 1
            # Convert dict to AgentResponse
            return AgentResponse(
                content=response_dict.get("content", ""),
                tool_calls=response_dict.get("tool_calls", []),
                done=response_dict.get("done", not bool(response_dict.get("tool_calls"))),
            )
        return AgentResponse(content="Done", tool_calls=[], done=True)

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        """Format tool results."""
        formatted = []
        for tool_call, result in zip(tool_calls, tool_results):
            formatted.append({"role": "tool", "name": tool_call.name, "content": str(result)})
        return formatted

    async def create_user_message(self, text: str) -> Any:
        """Create user message."""
        return {"role": "user", "content": text}


class TestMCPAgentExtended:
    """Extended tests for MCPAgent."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock MCP client."""
        client = MagicMock()
        client.get_all_active_sessions = MagicMock(return_value={})
        client.initialize = AsyncMock()
        client.list_tools = AsyncMock(return_value=[])
        client.call_tool = AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="Success")],
                isError=False,
            )
        )
        return client

    @pytest.fixture
    def agent_with_tools(self, mock_client):
        """Create agent with mock tools."""
        mock_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(name="screenshot", description="Take screenshot", inputSchema={}),
                types.Tool(name="click", description="Click at coordinates", inputSchema={}),
                types.Tool(name="type", description="Type text", inputSchema={}),
                types.Tool(name="bad_tool", description="A tool that fails", inputSchema={}),
            ]
        )
        return MockAgentExtended(mcp_client=mock_client)

    @pytest.mark.asyncio
    async def test_run_with_task_object(self, agent_with_tools):
        """Test running agent with TaskConfig object."""
        from hud.types import MCPToolResult

        task = TaskConfig(
            id="test_task",
            prompt="Click the button",
            mcp_config={"test_server": {"url": "http://localhost:8080"}},
            setup_tool={"name": "navigate", "arguments": {"url": "https://example.com"}},  # type: ignore[arg-type]
            evaluate_tool={"name": "check_result", "arguments": {}},  # type: ignore[arg-type]
        )

        # Set up responses
        agent_with_tools.responses = [
            {
                "role": "assistant",
                "content": "I'll click the button",
                "tool_calls": [MCPToolCall(name="click", arguments={"x": 100, "y": 200})],
            }
        ]

        # Mock the evaluation to return a reward
        agent_with_tools.mcp_client.call_tool = AsyncMock(
            side_effect=[
                # Setup tool
                MCPToolResult(
                    content=[types.TextContent(type="text", text="Navigated")],
                    isError=False,
                ),
                # Click tool
                MCPToolResult(
                    content=[types.TextContent(type="text", text="Clicked")],
                    isError=False,
                ),
                # Evaluate tool with reward
                MCPToolResult(
                    content=[types.TextContent(type="text", text="Success")],
                    isError=False,
                    structuredContent={"reward": 1.0},
                ),
            ]
        )

        result = await agent_with_tools.run(task)

        assert isinstance(result, Trace)
        assert result.reward == 1.0
        assert not result.isError
        assert result.done

    @pytest.mark.asyncio
    async def test_run_with_setup_error(self, agent_with_tools):
        """Test task execution with setup phase error."""
        from hud.types import MCPToolResult

        task = TaskConfig(
            id="test_task",
            prompt="Do something",
            mcp_config={"test_server": {"url": "http://localhost:8080"}},
            setup_tool={"name": "bad_setup", "arguments": {}},  # type: ignore[arg-type]
        )

        # Mock setup tool to fail
        agent_with_tools.mcp_client.call_tool = AsyncMock(
            return_value=MCPToolResult(
                content=[types.TextContent(type="text", text="Setup failed")],
                isError=True,
            )
        )

        result = await agent_with_tools.run(task)

        assert isinstance(result, Trace)
        assert result.isError
        # Error content is the string representation of the MCPToolResult list
        assert result.content is not None
        assert "Setup failed" in result.content
        assert "MCPToolResult" in result.content

    @pytest.mark.asyncio
    async def test_run_with_multiple_setup_tools(self, agent_with_tools):
        """Test task with multiple setup tools."""

        task = TaskConfig(
            id="test_task",
            prompt="Test multiple setup",
            mcp_config={"test_server": {"url": "http://localhost:8080"}},
            setup_tool=[
                MCPToolCall(name="setup1", arguments={}),
                MCPToolCall(name="setup2", arguments={}),
            ],
        )

        agent_with_tools.responses = [{"role": "assistant", "content": "Done", "tool_calls": []}]

        setup_calls = []
        agent_with_tools.mcp_client.call_tool = AsyncMock(
            side_effect=lambda name, arguments=None: setup_calls.append(name)
            or types.CallToolResult(
                content=[types.TextContent(type="text", text=f"{name} done")],
                isError=False,
            )
        )

        result = await agent_with_tools.run(task)

        assert "setup1" in setup_calls
        assert "setup2" in setup_calls
        assert not result.isError

    @pytest.mark.asyncio
    async def test_screenshot_capture_success(self, agent_with_tools):
        """Test successful screenshot capture."""
        # Initialize agent to make tools available
        await agent_with_tools.initialize()

        agent_with_tools.mcp_client.call_tool = AsyncMock(
            return_value=types.CallToolResult(
                content=[
                    types.ImageContent(type="image", data="base64_screenshot", mimeType="image/png")
                ],
                isError=False,
            )
        )

        screenshot = await agent_with_tools.capture_screenshot()
        assert screenshot == "base64_screenshot"

    @pytest.mark.asyncio
    async def test_screenshot_capture_failure(self, agent_with_tools):
        """Test screenshot capture failure."""
        agent_with_tools.mcp_client.call_tool = AsyncMock(
            side_effect=Exception("Screenshot failed")
        )

        screenshot = await agent_with_tools.capture_screenshot()
        assert screenshot is None

    @pytest.mark.asyncio
    async def test_extract_latest_screenshot(self, agent_with_tools):
        """Test extracting latest screenshot from tool results."""
        results = [
            types.CallToolResult(
                content=[
                    types.TextContent(type="text", text="Text content"),
                    types.ImageContent(type="image", data="screenshot1", mimeType="image/png"),
                ],
                isError=False,
            ),
            types.CallToolResult(
                content=[
                    types.ImageContent(type="image", data="screenshot2", mimeType="image/png")
                ],
                isError=False,
            ),
        ]

        screenshot = agent_with_tools.extract_latest_screenshot(results)
        assert screenshot == "screenshot2"

    @pytest.mark.asyncio
    async def test_extract_latest_screenshot_with_errors(self, agent_with_tools):
        """Test screenshot extraction skips error results."""
        results = [
            types.CallToolResult(
                content=[
                    types.ImageContent(type="image", data="good_screenshot", mimeType="image/png")
                ],
                isError=False,
            ),
            types.CallToolResult(
                content=[
                    types.ImageContent(type="image", data="bad_screenshot", mimeType="image/png")
                ],
                isError=True,
            ),
        ]

        screenshot = agent_with_tools.extract_latest_screenshot(results)
        assert screenshot == "good_screenshot"

    @pytest.mark.asyncio
    async def test_allowed_tools_filtering(self, mock_client):
        """Test that allowed_tools filters available tools."""
        mock_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(name="tool1", description="Tool 1", inputSchema={}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={}),
                types.Tool(name="tool3", description="Tool 3", inputSchema={}),
            ]
        )

        agent = MockAgentExtended(mcp_client=mock_client, allowed_tools=["tool1", "tool3"])
        await agent.initialize("test")

        available_names = [tool.name for tool in agent._available_tools]
        assert "tool1" in available_names
        assert "tool3" in available_names
        assert "tool2" not in available_names

    @pytest.mark.asyncio
    async def test_disallowed_tools_filtering(self, mock_client):
        """Test that disallowed_tools filters available tools."""
        mock_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(name="tool1", description="Tool 1", inputSchema={}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={}),
                types.Tool(name="tool3", description="Tool 3", inputSchema={}),
            ]
        )

        agent = MockAgentExtended(mcp_client=mock_client, disallowed_tools=["tool2"])
        await agent.initialize("test")

        available_names = [tool.name for tool in agent._available_tools]
        assert "tool1" in available_names
        assert "tool3" in available_names
        assert "tool2" not in available_names

    @pytest.mark.asyncio
    async def test_lifecycle_tools(self, mock_client):
        """Test lifecycle tools are called in run_prompt."""
        # Lifecycle tools are specified by name, not as objects
        agent = MockAgentExtended(
            mcp_client=mock_client,
            lifecycle_tools=["screenshot"],  # Use tool name
            responses=[{"role": "assistant", "content": "Done", "tool_calls": []}],
        )

        # Add screenshot tool to available tools
        mock_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(name="screenshot", description="Take screenshot", inputSchema={})
            ]
        )

        # Initialize to make tools available
        await agent.initialize()

        result = await agent.run("Test lifecycle", max_steps=1)
        assert not result.isError

    @pytest.mark.asyncio
    async def test_screenshot_history_management(self, agent_with_tools):
        """Test screenshot history is maintained within max limit."""
        agent_with_tools.max_screenshot_history = 2
        agent_with_tools.initial_screenshot = True

        # Mock screenshot capture
        agent_with_tools.capture_screenshot = AsyncMock(
            side_effect=lambda: f"screenshot_{len(agent_with_tools._screenshot_history)}"
        )

        # Set up responses with tool calls
        agent_with_tools.responses = [
            {
                "role": "assistant",
                "content": "Action 1",
                "tool_calls": [MCPToolCall(name="click", arguments={"x": 1, "y": 1})],
            },
            {
                "role": "assistant",
                "content": "Action 2",
                "tool_calls": [MCPToolCall(name="click", arguments={"x": 2, "y": 2})],
            },
            {
                "role": "assistant",
                "content": "Action 3",
                "tool_calls": [MCPToolCall(name="click", arguments={"x": 3, "y": 3})],
            },
        ]

        await agent_with_tools.run("Test screenshots", max_steps=3)

        # Should have max 2 screenshots in history
        assert len(agent_with_tools.screenshot_history) <= 2

    @pytest.mark.asyncio
    async def test_run_with_invalid_prompt_type(self, agent_with_tools):
        """Test run with invalid prompt type raises TypeError."""
        with pytest.raises(TypeError, match="prompt_or_task must be str or TaskConfig"):
            await agent_with_tools.run(123)  # Invalid type

    @pytest.mark.asyncio
    async def test_system_prompt_generation(self, agent_with_tools):
        """Test system prompt generation with custom and tool prompts."""
        # Initialize to populate available tools
        await agent_with_tools.initialize()

        agent_with_tools.custom_system_prompt = "Custom instructions."
        agent_with_tools.append_tool_system_prompt = True

        system_prompt = agent_with_tools.get_system_prompt()

        assert "Custom instructions." in system_prompt
        assert "You have access to the following tools:" in system_prompt
        assert "screenshot" in system_prompt

    @pytest.mark.asyncio
    async def test_system_prompt_without_tools(self, mock_client):
        """Test system prompt when append_tool_system_prompt is False."""
        agent = MockAgentExtended(
            mcp_client=mock_client,
            custom_system_prompt="Only custom.",
            append_tool_system_prompt=False,
        )

        system_prompt = agent.get_system_prompt()

        assert system_prompt == "Only custom."
        assert "Available tools:" not in system_prompt

    @pytest.mark.asyncio
    async def test_evaluate_phase_with_multiple_tools(self, agent_with_tools):
        """Test evaluation phase with multiple evaluation tools."""
        from hud.types import MCPToolResult

        task = TaskConfig(
            id="test_task",
            prompt="Test evaluation",
            mcp_config={"test_server": {"url": "http://localhost:8080"}},
            evaluate_tool=[
                MCPToolCall(name="eval1", arguments={}),
                MCPToolCall(name="eval2", arguments={"reward": True}),
            ],
        )

        agent_with_tools.responses = [{"role": "assistant", "content": "Done", "tool_calls": []}]

        eval_results = []
        agent_with_tools.mcp_client.call_tool = AsyncMock(
            side_effect=lambda name, arguments=None: eval_results.append(name)
            or MCPToolResult(
                content=[types.TextContent(type="text", text=f"{name} result")],
                isError=False,
                structuredContent={"reward": 0.5} if name == "eval1" else {"reward": 1.0},
            )
        )

        result = await agent_with_tools.run(task)

        assert "eval1" in eval_results
        assert "eval2" in eval_results
        assert result.reward == 0.5  # From eval1 (first evaluation tool)

    @pytest.mark.asyncio
    async def test_trace_population_on_error(self, agent_with_tools):
        """Test that trace is populated on task execution error."""

        task = TaskConfig(
            id="test_task",
            prompt="Test error",
            mcp_config={"test_server": {"url": "http://localhost:8080"}},
            setup_tool={"name": "failing_setup", "arguments": {}},  # type: ignore[arg-type]
        )

        # Make setup fail with exception
        agent_with_tools.mcp_client.call_tool = AsyncMock(side_effect=Exception("Setup explosion"))

        result = await agent_with_tools.run(task)

        assert result.isError
        # Error content is the string representation of the MCPToolResult list
        assert "Setup explosion" in result.content
        assert "MCPToolResult" in result.content
        assert result.done
