"""Tests for MCPAgent.run() with EvalContext."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from mcp import types

from hud.agents import MCPAgent
from hud.agents.base import BaseCreateParams
from hud.environment.router import ToolRouter
from hud.eval.context import EvalContext
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult


class MockConfig(BaseAgentConfig):
    model_name: str = "MockAgent"
    model: str = "mock-model"


class MockCreateParams(BaseCreateParams, MockConfig):
    pass


class MockMCPAgent(MCPAgent):
    """Mock agent for testing run()."""

    metadata: ClassVar[dict[str, Any] | None] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = MockConfig

    def __init__(self, **kwargs: Any) -> None:
        params = MockCreateParams(**kwargs)
        super().__init__(params)
        self._response = AgentResponse(content="Test response", tool_calls=[], done=True)

    def set_response(self, response: AgentResponse) -> None:
        self._response = response

    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        return self._response

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        return [{"role": "tool", "content": str(r)} for r in tool_results]

    async def get_system_messages(self) -> list[Any]:
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        return [{"type": "text", "text": getattr(b, "text")} for b in blocks if hasattr(b, "text")]


class MockEvalContext(EvalContext):
    """Mock EvalContext for testing - inherits from real EvalContext."""

    def __init__(self, prompt: str = "Test prompt", tools: list[types.Tool] | None = None) -> None:
        # Core attributes
        self.prompt = prompt
        self._tools = tools or [types.Tool(name="test_tool", description="Test", inputSchema={})]
        self._submitted: str | None = None
        self.reward: float | None = None
        self._initialized = True

        # Environment attributes
        self._router = ToolRouter()
        self._agent_include: list[str] | None = None
        self._agent_exclude: list[str] | None = None

        # EvalContext attributes
        self._task = None
        self.trace_id = "test-trace-id"
        self.eval_name = "test-eval"
        self.job_id: str | None = None
        self.group_id: str | None = None
        self.index = 0
        self.variants: dict[str, Any] = {}
        self.answer: str | None = None
        self.system_prompt: str | None = None
        self.error: BaseException | None = None
        self.metadata: dict[str, Any] = {}
        self.results: list[Any] = []
        self._is_summary = False

    def as_tools(self) -> list[types.Tool]:
        return self._tools

    @property
    def has_scenario(self) -> bool:
        return True

    async def list_tools(self) -> list[types.Tool]:
        return self._tools

    async def call_tool(self, call: Any, /, **kwargs: Any) -> MCPToolResult:
        # Handle tuple format (name, args)
        if isinstance(call, tuple):
            name = call[0]
        elif hasattr(call, "name"):
            name = call.name
        else:
            name = str(call)
        return MCPToolResult(
            content=[types.TextContent(type="text", text=f"Result from {name}")],
            isError=False,
        )

    async def submit(self, answer: str) -> None:
        self._submitted = answer


class TestRun:
    """Tests for MCPAgent.run() with EvalContext."""

    @pytest.mark.asyncio
    async def test_run_basic(self) -> None:
        """Test basic run() flow."""
        ctx = MockEvalContext(prompt="Do the task")
        agent = MockMCPAgent()

        result = await agent.run(ctx)

        assert result.done
        assert result.content == "Test response"
        assert ctx._submitted == "Test response"

    @pytest.mark.asyncio
    async def test_run_no_prompt_raises(self) -> None:
        """Test run() raises when prompt is not set."""
        ctx = MockEvalContext(prompt="")
        agent = MockMCPAgent()

        with pytest.raises(ValueError, match="prompt is not set"):
            await agent.run(ctx)

    @pytest.mark.asyncio
    async def test_run_wrong_type_raises(self) -> None:
        """Test run() raises TypeError for non-EvalContext."""
        agent = MockMCPAgent()

        with pytest.raises(TypeError, match="must be EvalContext"):
            await agent.run("not an eval context")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_run_clears_ctx(self) -> None:
        """Test run() clears ctx after completion."""
        ctx = MockEvalContext(prompt="Do the task")
        agent = MockMCPAgent()

        await agent.run(ctx)
        assert agent.ctx is None

    @pytest.mark.asyncio
    async def test_run_no_submit_on_empty_content(self) -> None:
        """Test run() doesn't submit when content is empty."""
        ctx = MockEvalContext(prompt="Do the task")
        agent = MockMCPAgent()
        agent.set_response(AgentResponse(content="", tool_calls=[], done=True))

        await agent.run(ctx)
        assert ctx._submitted is None

    @pytest.mark.asyncio
    async def test_run_initializes_tools(self) -> None:
        """Test run() initializes tools from context."""
        ctx = MockEvalContext(
            prompt="Do the task",
            tools=[
                types.Tool(name="tool1", description="Tool 1", inputSchema={}),
                types.Tool(name="tool2", description="Tool 2", inputSchema={}),
            ],
        )
        agent = MockMCPAgent()

        await agent.run(ctx)

        assert agent._initialized
        # After cleanup, ctx is None but tools were discovered
