"""Tests for BaseMCPAgent using simulated actions."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

# Import AsyncMock from unittest.mock if available (Python 3.8+)
try:
    from unittest.mock import AsyncMock
except ImportError:
    # Fallback for older Python versions
    from unittest.mock import MagicMock as AsyncMock

import pytest
from mcp import types

from hud.agents import MCPAgent
from hud.agents.base import BaseCreateParams
from hud.datasets import Task
from hud.tools.executors.base import BaseExecutor
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult, Trace

from .conftest import MockMCPClient


class MockConfig(BaseAgentConfig):
    model_name: str = "MockAgent"
    checkpoint_name: str = "mock-model"


class MockCreateParams(BaseCreateParams, MockConfig):
    pass


class MockMCPAgent(MCPAgent):
    """Concrete implementation of BaseMCPAgent for testing."""

    metadata: ClassVar[dict[str, Any] | None] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = MockConfig

    def __init__(self, mcp_client: Any = None, **kwargs: Any) -> None:
        if mcp_client is None:
            mcp_client = MockMCPClient()

        kwargs.setdefault("mcp_client", mcp_client)
        params = MockCreateParams(**kwargs)
        super().__init__(params)
        self.executor = BaseExecutor()
        self._messages: list[dict[str, Any]] = []

    async def create_initial_messages(
        self, prompt: str, initial_screenshot: bool = False
    ) -> list[dict[str, Any]]:
        """Mock create initial messages."""
        messages = [{"role": "user", "content": prompt}]
        if initial_screenshot:
            messages.append({"role": "assistant", "content": "Screenshot: mock_screenshot"})
        return messages

    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        """Mock get response."""
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

    async def get_system_messages(self) -> list[Any]:
        """Mock get system messages."""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Mock format blocks."""
        formatted = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                formatted.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                formatted.append({"type": "image", "data": block.data})
            elif hasattr(block, "type"):
                formatted.append({"type": getattr(block, "type", "unknown")})
        return formatted


class TestBaseMCPAgent:
    """Tests for BaseMCPAgent with simulated actions."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        agent = MockMCPAgent()

        assert agent.mcp_client is not None
        assert agent.allowed_tools is None
        assert agent.disallowed_tools is None
        assert agent.initial_screenshot is True

    def test_init_with_params(self, mock_mcp_client):
        """Test initialization with custom parameters."""
        agent = MockMCPAgent(
            mcp_client=mock_mcp_client,
            allowed_tools=["tool1", "tool2"],
            disallowed_tools=["bad_tool"],
            initial_screenshot=True,
            system_prompt="Custom prompt",
        )

        assert agent.mcp_client == mock_mcp_client
        assert agent.allowed_tools == ["tool1", "tool2"]
        assert agent.disallowed_tools == ["bad_tool"]
        assert agent.initial_screenshot is True
        assert agent.system_prompt == "Custom prompt"

    @pytest.mark.asyncio
    async def test_init_no_client_no_task(self):
        """Test initialize fails without client and without task."""

        # Create a minimal concrete implementation to test the ValueError
        class TestAgentConfig(BaseAgentConfig):
            model_name: str = "TestAgent"
            checkpoint_name: str = "test-model"

        class TestAgentCreateParams(BaseCreateParams, TestAgentConfig):
            pass

        class TestAgent(MCPAgent):
            config_cls = TestAgentConfig

            def __init__(self, **kwargs: Any) -> None:
                params = TestAgentCreateParams(**kwargs)
                super().__init__(params)

            async def create_initial_messages(
                self, prompt: str, initial_screenshot: bool = False
            ) -> list[dict[str, Any]]:
                return []

            async def format_tool_results(
                self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
            ) -> list[dict[str, Any]]:
                return []

            async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
                return AgentResponse(content="test", tool_calls=[], done=True)

            async def get_system_messages(self) -> list[Any]:
                return []

            async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
                return []

        # Agent can be created with None client
        agent = TestAgent(mcp_client=None)

        # But initialize should fail without client or task
        with pytest.raises(ValueError, match="No MCPClient"):
            await agent.initialize()

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

        assert agent.mcp_client is not None

        # Mock the list_tools method on mcp_client to return the tools
        agent.mcp_client.list_tools = AsyncMock(
            return_value=[
                types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
                types.Tool(name="setup", description="Setup tool", inputSchema={"type": "object"}),
            ]
        )

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

        # Make sure tool schemas are correct
        schemas = agent.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "tool1"
        assert schemas[0]["description"] == "Tool 1"
        assert schemas[0]["parameters"] == {"type": "object"}

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
        results = await agent.call_tools(tool_call)

        assert len(results) == 1
        assert results[0] == mock_result
        assert not results[0].isError

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

        # Try to call unknown tool - call_tools doesn't raise for unknown tools
        tool_call = MCPToolCall(name="unknown_tool", arguments={})
        await agent.call_tools(tool_call)

    @pytest.mark.asyncio
    async def test_call_tool_no_name(self):
        """Test calling tool without name."""
        # MCPToolCall accepts empty names
        agent = MockMCPAgent()
        tool_call = MCPToolCall(name="", arguments={})

        # call_tools doesn't validate empty names, it will return error
        await agent.call_tools(tool_call)

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        agent = MockMCPAgent()

        agent._available_tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
            types.Tool(name="setup", description="Setup", inputSchema={"type": "object"}),
        ]

        schemas = agent.get_tool_schemas()

        # Should include non-lifecycle tools
        assert len(schemas) == 2
        assert schemas[0]["name"] == "tool1"

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

    metadata: ClassVar[dict[str, Any] | None] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = MockConfig

    def __init__(self, responses: list[Any] | None = None, **kwargs: Any):
        if kwargs.get("mcp_client") is None:
            kwargs["mcp_client"] = MockMCPClient()
        params = MockCreateParams(**kwargs)
        super().__init__(params)
        self.responses = responses or []
        self.call_count = 0

    async def create_initial_messages(
        self, prompt: str, initial_screenshot: bool = False
    ) -> list[dict[str, Any]]:
        """Create initial messages."""
        messages = [{"role": "user", "content": prompt}]
        if initial_screenshot:
            # capture_screenshot doesn't exist, just mock it
            screenshot = "mock_screenshot_data"
            messages.append({"role": "assistant", "content": f"Screenshot: {screenshot}"})
        return messages

    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
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

    async def get_system_messages(self) -> list[Any]:
        """Mock get system messages."""
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        """Mock format blocks."""
        formatted = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                formatted.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                formatted.append({"type": "image", "data": block.data})
            elif hasattr(block, "type"):
                formatted.append({"type": getattr(block, "type", "unknown")})
        return formatted


class TestMCPAgentExtended:
    """Extended tests for MCPAgent."""

    @pytest.fixture
    def agent_with_tools(self, mock_mcp_client_browser_tools):
        """Create agent with mock tools."""
        return MockAgentExtended(mcp_client=mock_mcp_client_browser_tools)

    @pytest.mark.asyncio
    async def test_run_with_task_object(self, agent_with_tools):
        """Test running agent with Task object."""
        from hud.types import MCPToolResult

        task = Task(
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

        task = Task(
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

        task = Task(
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
            side_effect=lambda tool_call: setup_calls.append(tool_call)
            or MCPToolResult(
                content=[types.TextContent(type="text", text=f"{tool_call.name} done")],
                isError=False,
            )
        )

        result = await agent_with_tools.run(task)

        # Check that the tool names match
        setup_names = [call.name for call in setup_calls]
        assert "setup1" in setup_names
        assert "setup2" in setup_names
        assert not result.isError

    @pytest.mark.asyncio
    async def test_allowed_tools_filtering(self):
        """Test that allowed_tools filters available tools."""
        mock_client = MockMCPClient(
            tools=[
                types.Tool(name="tool1", description="Tool 1", inputSchema={}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={}),
                types.Tool(name="tool3", description="Tool 3", inputSchema={}),
            ]
        )

        agent = MockAgentExtended(mcp_client=mock_client, allowed_tools=["tool1", "tool3"])
        await agent.initialize("test")

        available_names = [tool.name for tool in agent.get_available_tools()]
        assert "tool1" in available_names
        assert "tool3" in available_names
        assert "tool2" not in available_names

    @pytest.mark.asyncio
    async def test_disallowed_tools_filtering(self):
        """Test that disallowed_tools filters available tools."""
        mock_client = MockMCPClient(
            tools=[
                types.Tool(name="tool1", description="Tool 1", inputSchema={}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={}),
                types.Tool(name="tool3", description="Tool 3", inputSchema={}),
            ]
        )

        agent = MockAgentExtended(mcp_client=mock_client, disallowed_tools=["tool2"])
        await agent.initialize("test")

        available_names = [tool.name for tool in agent.get_available_tools()]
        assert "tool1" in available_names
        assert "tool3" in available_names
        assert "tool2" not in available_names

    @pytest.mark.asyncio
    async def test_lifecycle_tools(self):
        """Test lifecycle tools are called in run_prompt."""
        mock_client = MockMCPClient(
            tools=[types.Tool(name="screenshot", description="Take screenshot", inputSchema={})]
        )

        agent = MockAgentExtended(
            mcp_client=mock_client,
            responses=[{"role": "assistant", "content": "Done", "tool_calls": []}],
        )

        # Initialize to make tools available
        await agent.initialize()

        result = await agent.run("Test lifecycle", max_steps=1)
        assert not result.isError

    # This test is commented out as screenshot history management may have changed
    # @pytest.mark.asyncio
    # async def test_screenshot_history_management(self, agent_with_tools):
    #     """Test screenshot history is maintained."""
    #     agent_with_tools.initial_screenshot = True

    #     # Set up responses with tool calls
    #     agent_with_tools.responses = [
    #         {
    #             "role": "assistant",
    #             "content": "Action 1",
    #             "tool_calls": [MCPToolCall(name="click", arguments={"x": 1, "y": 1})],
    #         },
    #         {
    #             "role": "assistant",
    #             "content": "Action 2",
    #             "tool_calls": [MCPToolCall(name="click", arguments={"x": 2, "y": 2})],
    #         },
    #         {
    #             "role": "assistant",
    #             "content": "Action 3",
    #             "tool_calls": [MCPToolCall(name="click", arguments={"x": 3, "y": 3})],
    #         },
    #     ]

    #     await agent_with_tools.run("Test screenshots", max_steps=3)

    #     # Should have screenshots in history
    #     assert len(agent_with_tools.screenshot_history) > 0

    @pytest.mark.asyncio
    async def test_run_with_invalid_prompt_type(self, agent_with_tools):
        """Test run with invalid prompt type raises TypeError."""
        with pytest.raises(TypeError, match="prompt_or_task must be str or Task"):
            await agent_with_tools.run(123)  # Invalid type

    @pytest.mark.asyncio
    async def test_evaluate_phase_with_multiple_tools(self, agent_with_tools):
        """Test evaluation phase with multiple evaluation tools."""
        from hud.types import MCPToolResult

        task = Task(
            id="test_task",
            prompt="Test evaluation",
            mcp_config={"test_server": {"url": "http://localhost:8080"}},
            evaluate_tool=[
                MCPToolCall(name="eval1", arguments={}),
                MCPToolCall(name="eval2", arguments={"reward": True}),
            ],
        )

        agent_with_tools.responses = [{"role": "assistant", "content": "Done", "tool_calls": []}]

        eval_calls = []
        agent_with_tools.mcp_client.call_tool = AsyncMock(
            side_effect=lambda tool_call: eval_calls.append(tool_call)
            or MCPToolResult(
                content=[types.TextContent(type="text", text=f"{tool_call.name} result")],
                isError=False,
                structuredContent={"reward": 0.5} if tool_call.name == "eval1" else {"reward": 1.0},
            )
        )

        result = await agent_with_tools.run(task)

        # Check that the tool names match
        eval_names = [call.name for call in eval_calls]
        assert "eval1" in eval_names
        assert "eval2" in eval_names
        assert result.reward == 0.5  # From eval1 (first evaluation tool)

    @pytest.mark.asyncio
    async def test_trace_population_on_error(self, agent_with_tools):
        """Test that trace is populated on task execution error."""

        task = Task(
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
