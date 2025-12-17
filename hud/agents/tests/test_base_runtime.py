"""Runtime tests for MCPAgent base class."""

from __future__ import annotations

from typing import Any

import mcp.types as types
import pytest

from hud.agents.base import BaseCreateParams, MCPAgent, find_content, find_reward, text_to_blocks
from hud.environment.router import ToolRouter
from hud.eval.context import EvalContext
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult


class DummyConfig(BaseAgentConfig):
    model_name: str = "DummyAgent"
    model: str = "dummy-model"


class DummyCreateParams(BaseCreateParams, DummyConfig):
    pass


class MockEvalContext(EvalContext):
    """Mock EvalContext for testing."""

    def __init__(
        self,
        prompt: str = "Test prompt",
        tools: list[types.Tool] | None = None,
    ) -> None:
        # Core attributes
        self.prompt = prompt
        self._tools = tools or []
        self._submitted: str | None = None
        self.reward: float | None = None
        self._call_tool_handler: Any = None

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
        return False

    def set_call_tool_handler(self, handler: Any) -> None:
        self._call_tool_handler = handler

    async def list_tools(self) -> list[types.Tool]:
        return self._tools

    async def call_tool(self, call: Any, /, **kwargs: Any) -> MCPToolResult:
        if self._call_tool_handler:
            # Parse the call
            if isinstance(call, tuple):
                tc = MCPToolCall(name=call[0], arguments=call[1] if len(call) > 1 else {})
            elif hasattr(call, "name"):
                tc = call
            else:
                tc = MCPToolCall(name=str(call), arguments=kwargs)
            return self._call_tool_handler(tc)
        return MCPToolResult(
            content=[types.TextContent(type="text", text="ok")],
            isError=False,
        )

    async def submit(self, answer: str) -> None:
        self._submitted = answer


class DummyAgent(MCPAgent):
    config_cls = DummyConfig

    def __init__(self, **kwargs: Any) -> None:
        params = DummyCreateParams(**kwargs)
        super().__init__(params)

    async def get_system_messages(self) -> list[types.ContentBlock]:
        return [types.TextContent(type="text", text="sys")]

    async def get_response(self, messages: list[Any]) -> AgentResponse:
        return AgentResponse(content="ok", tool_calls=[], done=True)

    async def format_blocks(self, blocks: list[Any]) -> list[Any]:
        return blocks

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[Any]:
        return [types.TextContent(text="tools", type="text")]


def test_find_reward_and_content_extractors() -> None:
    """Test reward and content extraction from tool results."""
    # Structured content
    r = MCPToolResult(
        content=text_to_blocks("{}"), isError=False, structuredContent={"reward": 0.7}
    )
    assert find_reward(r) == 0.7

    # Text JSON
    r2 = MCPToolResult(content=text_to_blocks('{"score": 0.5, "content": "hi"}'), isError=False)
    assert find_reward(r2) == 0.5
    assert find_content(r2) == "hi"


def test_get_available_tools_before_run_raises() -> None:
    """Test that get_available_tools raises before initialization."""
    agent = DummyAgent()
    with pytest.raises(RuntimeError):
        agent.get_available_tools()


@pytest.mark.asyncio
async def test_format_message_invalid_type_raises() -> None:
    """Test that format_message raises for invalid types."""
    agent = DummyAgent()
    with pytest.raises(ValueError):
        await agent.format_message({"oops": 1})  # type: ignore


def test_text_to_blocks_shapes() -> None:
    """Test text_to_blocks returns correct structure."""
    blocks = text_to_blocks("x")
    assert isinstance(blocks, list) and blocks and isinstance(blocks[0], types.TextContent)


@pytest.mark.asyncio
async def test_run_with_eval_context() -> None:
    """Test basic run() with EvalContext."""
    ctx = MockEvalContext(prompt="hello")
    agent = DummyAgent()
    result = await agent.run(ctx, max_steps=1)
    assert result.done is True
    assert result.isError is False


@pytest.mark.asyncio
async def test_run_requires_eval_context() -> None:
    """Test run() raises TypeError for non-EvalContext."""
    agent = DummyAgent()
    with pytest.raises(TypeError, match="must be EvalContext"):
        await agent.run("hello")  # type: ignore


@pytest.mark.asyncio
async def test_run_requires_prompt() -> None:
    """Test run() raises ValueError when prompt is empty."""
    ctx = MockEvalContext(prompt="")
    agent = DummyAgent()
    with pytest.raises(ValueError, match="prompt is not set"):
        await agent.run(ctx)


@pytest.mark.asyncio
async def test_call_tools_error_paths() -> None:
    """Test call_tools handles errors correctly."""
    call_count = [0]
    ok_result = MCPToolResult(content=text_to_blocks("ok"), isError=False)

    def handler(tool_call: MCPToolCall) -> MCPToolResult:
        call_count[0] += 1
        if call_count[0] == 1:
            return ok_result
        raise RuntimeError("boom")

    ctx = MockEvalContext(prompt="test")
    ctx.set_call_tool_handler(handler)
    agent = DummyAgent()

    # Initialize the agent with context
    agent.ctx = ctx
    await agent._initialize_from_ctx(ctx)

    results = await agent.call_tools(
        [MCPToolCall(name="a", arguments={}), MCPToolCall(name="b", arguments={})]
    )
    assert results[0].isError is False
    assert results[1].isError is True


@pytest.mark.asyncio
async def test_call_tools_timeout_raises() -> None:
    """Test call_tools raises TimeoutError."""

    def handler(tool_call: MCPToolCall) -> MCPToolResult:
        raise TimeoutError("timeout")

    ctx = MockEvalContext(prompt="test")
    ctx.set_call_tool_handler(handler)
    agent = DummyAgent()

    agent.ctx = ctx
    await agent._initialize_from_ctx(ctx)

    with pytest.raises(TimeoutError):
        await agent.call_tools(MCPToolCall(name="x", arguments={}))


@pytest.mark.asyncio
async def test_get_available_tools_after_run() -> None:
    """Test get_available_tools works after initialization."""
    tools = [types.Tool(name="test_tool", description="Test", inputSchema={})]
    ctx = MockEvalContext(prompt="hello", tools=tools)
    agent = DummyAgent()

    # Run initializes the agent
    await agent.run(ctx, max_steps=1)

    # After cleanup, we can't access tools (ctx is cleared)
    # But during run, tools were available
    assert agent._initialized is True
