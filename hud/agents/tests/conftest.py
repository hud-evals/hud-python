"""Shared test fixtures for agent tests."""

from __future__ import annotations

from typing import Any

import pytest
from mcp import types

from hud.environment.router import ToolRouter
from hud.eval.context import EvalContext
from hud.types import MCPToolCall, MCPToolResult


class MockEvalContext(EvalContext):
    """Mock EvalContext for testing agents.

    This provides a minimal EvalContext implementation that can be used
    to test agent initialization and tool calling without a real environment.
    """

    def __init__(
        self,
        prompt: str = "Test prompt",
        tools: list[types.Tool] | None = None,
        call_tool_handler: Any = None,
    ) -> None:
        # Core attributes
        self.prompt = prompt
        self._tools = tools or []
        self._submitted: str | None = None
        self.reward: float | None = None
        self._call_tool_handler = call_tool_handler
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []

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

    async def list_tools(self) -> list[types.Tool]:
        return self._tools

    async def call_tool(self, call: Any, /, **kwargs: Any) -> MCPToolResult:
        # Parse the call
        if isinstance(call, tuple):
            name, args = call[0], call[1] if len(call) > 1 else {}
        elif hasattr(call, "name"):
            name, args = call.name, getattr(call, "arguments", {}) or {}
        else:
            name, args = str(call), kwargs

        self.tool_calls.append((name, args))

        if self._call_tool_handler:
            tc = MCPToolCall(name=name, arguments=args)
            return self._call_tool_handler(tc)

        return MCPToolResult(
            content=[types.TextContent(type="text", text=f"Result from {name}")],
            isError=False,
        )

    async def submit(self, answer: str) -> None:
        self._submitted = answer


@pytest.fixture
def mock_eval_context() -> MockEvalContext:
    """Create a basic mock EvalContext."""
    return MockEvalContext()


@pytest.fixture
def mock_eval_context_with_tools() -> MockEvalContext:
    """Create a mock EvalContext with test tools."""
    return MockEvalContext(
        tools=[
            types.Tool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {}},
            )
        ]
    )


@pytest.fixture
def mock_eval_context_computer() -> MockEvalContext:
    """Create a mock EvalContext with computer tool."""
    return MockEvalContext(
        tools=[
            types.Tool(
                name="computer",
                description="Computer use tool",
                inputSchema={"type": "object"},
            )
        ]
    )


@pytest.fixture
def mock_eval_context_browser_tools() -> MockEvalContext:
    """Create a mock EvalContext with browser-like tools."""
    return MockEvalContext(
        tools=[
            types.Tool(name="screenshot", description="Take screenshot", inputSchema={}),
            types.Tool(name="click", description="Click at coordinates", inputSchema={}),
            types.Tool(name="type", description="Type text", inputSchema={}),
        ]
    )
