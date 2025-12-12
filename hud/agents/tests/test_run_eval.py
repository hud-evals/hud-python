"""Tests for run_eval and EnvironmentClient."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from mcp import types

from hud.agents import MCPAgent
from hud.agents.base import BaseCreateParams
from hud.clients.environment import EnvironmentClient
from hud.eval.context import EvalContext
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult


class MockConfig(BaseAgentConfig):
    model_name: str = "MockAgent"
    checkpoint_name: str = "mock-model"


class MockCreateParams(BaseCreateParams, MockConfig):
    pass


class MockMCPAgent(MCPAgent):
    """Mock agent for testing run_eval."""

    metadata: ClassVar[dict[str, Any] | None] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = MockConfig

    def __init__(self, **kwargs: Any) -> None:
        params = MockCreateParams(**kwargs)
        super().__init__(params)
        self._response = AgentResponse(content="Test response", tool_calls=[], done=True)

    def set_response(self, response: AgentResponse) -> None:
        self._response = response

    async def create_initial_messages(
        self, prompt: str, initial_screenshot: bool = False
    ) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt}]

    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        return self._response

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        return [{"role": "tool", "content": str(r)} for r in tool_results]

    async def create_user_message(self, text: str) -> Any:
        return {"role": "user", "content": text}

    async def get_system_messages(self) -> list[Any]:
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        return [{"type": "text", "text": b.text} for b in blocks if hasattr(b, "text")]


class MockEvalContext(EvalContext):
    """Mock EvalContext for testing - inherits from real EvalContext."""

    def __init__(self, prompt: str = "Test prompt", tools: list[types.Tool] | None = None) -> None:
        # Skip parent __init__, just set what we need
        self.prompt = prompt
        self._tools = tools or [
            types.Tool(name="test_tool", description="Test", inputSchema={})
        ]
        self._submitted: str | None = None
        self.reward: float | None = None

    async def list_tools(self) -> list[types.Tool]:
        return self._tools

    async def call_tool(self, name: str, **kwargs: Any) -> MCPToolResult:
        return MCPToolResult(
            content=[types.TextContent(type="text", text=f"Result from {name}")],
            isError=False,
        )

    async def submit(self, answer: str) -> None:
        self._submitted = answer


class TestEnvironmentClient:
    """Tests for EnvironmentClient adapter."""

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Test client initialization."""
        ctx = MockEvalContext()
        client = EnvironmentClient(ctx)

        assert not client.is_connected
        await client.initialize()
        assert client.is_connected

    @pytest.mark.asyncio
    async def test_list_tools(self) -> None:
        """Test listing tools through adapter."""
        ctx = MockEvalContext()
        client = EnvironmentClient(ctx)

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_call_tool(self) -> None:
        """Test calling tools through adapter."""
        ctx = MockEvalContext()
        client = EnvironmentClient(ctx)

        result = await client.call_tool(MCPToolCall(name="test_tool", arguments={}))
        assert not result.isError
        assert len(result.content) == 1

    @pytest.mark.asyncio
    async def test_mcp_config_empty(self) -> None:
        """Test mcp_config is empty for environment clients."""
        ctx = MockEvalContext()
        client = EnvironmentClient(ctx)
        assert client.mcp_config == {}

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test shutdown resets initialized state."""
        ctx = MockEvalContext()
        client = EnvironmentClient(ctx)

        await client.initialize()
        assert client.is_connected

        await client.shutdown()
        assert not client.is_connected


class TestRunEval:
    """Tests for MCPAgent.run_eval()."""

    @pytest.mark.asyncio
    async def test_run_eval_basic(self) -> None:
        """Test basic run_eval flow."""
        ctx = MockEvalContext(prompt="Do the task")
        agent = MockMCPAgent()

        result = await agent.run_eval(ctx)

        assert result.done
        assert result.content == "Test response"
        assert ctx._submitted == "Test response"

    @pytest.mark.asyncio
    async def test_run_eval_no_prompt_raises(self) -> None:
        """Test run_eval raises when prompt is not set."""
        ctx = MockEvalContext(prompt="")
        agent = MockMCPAgent()

        with pytest.raises(ValueError, match="prompt is not set"):
            await agent.run_eval(ctx)

    @pytest.mark.asyncio
    async def test_run_eval_wrong_type_raises(self) -> None:
        """Test run_eval raises TypeError for non-EvalContext."""
        agent = MockMCPAgent()

        with pytest.raises(TypeError, match="must be EvalContext"):
            await agent.run_eval("not an eval context")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_run_eval_clears_client(self) -> None:
        """Test run_eval clears mcp_client after completion."""
        ctx = MockEvalContext(prompt="Do the task")
        agent = MockMCPAgent()

        await agent.run_eval(ctx)
        assert agent.mcp_client is None

    @pytest.mark.asyncio
    async def test_run_eval_no_submit_on_empty_content(self) -> None:
        """Test run_eval doesn't submit when content is empty."""
        ctx = MockEvalContext(prompt="Do the task")
        agent = MockMCPAgent()
        agent.set_response(AgentResponse(content="", tool_calls=[], done=True))

        await agent.run_eval(ctx)
        assert ctx._submitted is None
