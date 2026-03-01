"""Tests for MCPAgent base class with v5 EvalContext pattern."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from mcp import types

from hud.agents import MCPAgent
from hud.agents.base import BaseCreateParams
from hud.environment.router import ToolRouter
from hud.eval.context import EvalContext
from hud.types import AgentType, BaseAgentConfig, InferenceResult, MCPToolCall, MCPToolResult


class MockConfig(BaseAgentConfig):
    model_name: str = "MockAgent"
    model: str = "mock-model"


class MockCreateParams(BaseCreateParams, MockConfig):
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
        self._tools = tools or [
            types.Tool(name="test_tool", description="A test tool", inputSchema={}),
            types.Tool(name="another_tool", description="Another tool", inputSchema={}),
        ]
        self._submitted: str | dict[str, Any] | None = None
        self.reward: float | None = None
        self._tool_calls: list[tuple[str, dict[str, Any]]] = []

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
        self.answer: str | dict[str, Any] | None = None
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
        # Parse the call
        if isinstance(call, tuple):
            name, args = call[0], call[1] if len(call) > 1 else {}
        elif hasattr(call, "name"):
            name, args = call.name, getattr(call, "arguments", {}) or {}
        else:
            name, args = str(call), kwargs
        self._tool_calls.append((name, args))
        return MCPToolResult(
            content=[types.TextContent(type="text", text=f"Result from {name}")],
            isError=False,
        )

    async def submit(self, answer: str | dict[str, Any]) -> None:
        self._submitted = answer


class MockMCPAgent(MCPAgent):
    """Concrete implementation of MCPAgent for testing."""

    metadata: ClassVar[dict[str, Any] | None] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = MockConfig

    @classmethod
    def agent_type(cls) -> AgentType:
        """Return the AgentType for the mock agent."""
        return AgentType.INTEGRATION_TEST

    def __init__(self, **kwargs: Any) -> None:
        params = MockCreateParams(**kwargs)
        super().__init__(params)
        self._response = InferenceResult(content="Mock response", tool_calls=[], done=True)

    def set_response(self, response: InferenceResult) -> None:
        self._response = response

    async def get_response(self, messages: list[dict[str, Any]]) -> InferenceResult:
        return self._response

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        formatted = []
        for tool_call, result in zip(tool_calls, tool_results, strict=True):
            formatted.append({"role": "tool", "name": tool_call.name, "content": str(result)})
        return formatted

    async def get_system_messages(self) -> list[Any]:
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[Any]:
        return [{"type": "text", "text": getattr(b, "text", "")} for b in blocks]


class TestMCPAgentInit:
    """Tests for MCPAgent initialization."""

    def test_init_defaults(self) -> None:
        """Test agent initializes with default config."""
        agent = MockMCPAgent()
        assert agent.ctx is None
        assert agent._initialized is False
        assert agent.system_prompt is None

    def test_init_with_system_prompt(self) -> None:
        """Test agent with custom system prompt."""
        agent = MockMCPAgent(system_prompt="Custom prompt")
        assert agent.system_prompt == "Custom prompt"


class TestMCPAgentRun:
    """Tests for MCPAgent.run() with EvalContext."""

    @pytest.mark.asyncio
    async def test_run_basic(self) -> None:
        """Test basic run flow with EvalContext."""
        ctx = MockEvalContext(prompt="Do something")
        agent = MockMCPAgent()

        result = await agent.run(ctx)

        assert result.done is True
        assert result.content == "Mock response"
        assert ctx._submitted == "Mock response"

    @pytest.mark.asyncio
    async def test_run_initializes_agent(self) -> None:
        """Test run() initializes the agent with context."""
        ctx = MockEvalContext(prompt="Do something")
        agent = MockMCPAgent()

        assert not agent._initialized
        await agent.run(ctx)
        assert agent._initialized

    @pytest.mark.asyncio
    async def test_run_discovers_tools(self) -> None:
        """Test run() discovers tools from context."""
        tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={}),
            types.Tool(name="tool2", description="Tool 2", inputSchema={}),
        ]
        ctx = MockEvalContext(prompt="Do something", tools=tools)
        agent = MockMCPAgent()

        # We need to check tools before cleanup
        # Store a reference to check
        discovered_tools = []

        original_run = agent._run_context

        async def capture_tools(*args: Any, **kwargs: Any) -> Any:
            discovered_tools.extend(agent.get_available_tools())
            return await original_run(*args, **kwargs)

        agent._run_context = capture_tools  # type: ignore
        await agent.run(ctx)

        assert len(discovered_tools) == 2
        assert discovered_tools[0].name == "tool1"
        assert discovered_tools[1].name == "tool2"

    @pytest.mark.asyncio
    async def test_run_requires_eval_context(self) -> None:
        """Test run() raises TypeError for non-EvalContext."""
        agent = MockMCPAgent()

        with pytest.raises(TypeError, match="must be EvalContext"):
            await agent.run("not a context")  # type: ignore

    @pytest.mark.asyncio
    async def test_run_requires_prompt(self) -> None:
        """Test run() raises ValueError when prompt is empty."""
        ctx = MockEvalContext(prompt="")
        agent = MockMCPAgent()

        with pytest.raises(ValueError, match="prompt is not set"):
            await agent.run(ctx)

    @pytest.mark.asyncio
    async def test_run_clears_context_after(self) -> None:
        """Test run() clears ctx after completion."""
        ctx = MockEvalContext(prompt="Do something")
        agent = MockMCPAgent()

        await agent.run(ctx)
        assert agent.ctx is None

    @pytest.mark.asyncio
    async def test_run_no_submit_on_empty_content(self) -> None:
        """Test run() doesn't submit when content is empty."""
        ctx = MockEvalContext(prompt="Do something")
        agent = MockMCPAgent()
        agent.set_response(InferenceResult(content="", tool_calls=[], done=True))

        await agent.run(ctx)
        assert ctx._submitted is None


class TestMCPAgentToolCalling:
    """Tests for tool calling through context."""

    @pytest.mark.asyncio
    async def test_call_tools_uses_context(self) -> None:
        """Test call_tools routes through ctx.call_tool."""
        ctx = MockEvalContext(prompt="Do something")
        agent = MockMCPAgent()

        # Bind context manually
        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        # Call a tool
        results = await agent.call_tools(MCPToolCall(name="test_tool", arguments={"arg": "value"}))

        assert len(results) == 1
        assert not results[0].isError
        assert ("test_tool", {"arg": "value"}) in ctx._tool_calls

    @pytest.mark.asyncio
    async def test_call_tools_without_context_raises(self) -> None:
        """Test call_tools raises when no context bound."""
        agent = MockMCPAgent()

        with pytest.raises(ValueError, match="not bound to context"):
            await agent.call_tools(MCPToolCall(name="test_tool", arguments={}))


class TestMCPAgentRequiredTools:
    """Tests for required_tools validation."""

    @pytest.mark.asyncio
    async def test_missing_required_tools_raises(self) -> None:
        """Test run() raises when required tools are missing."""

        class AgentWithRequiredTools(MockMCPAgent):
            required_tools: ClassVar[list[str]] = ["must_have_tool"]

        ctx = MockEvalContext(prompt="Do something", tools=[])
        agent = AgentWithRequiredTools()

        with pytest.raises(ValueError, match="Required tools are missing"):
            await agent.run(ctx)

    @pytest.mark.asyncio
    async def test_required_tools_present_succeeds(self) -> None:
        """Test run() succeeds when required tools are present."""

        class AgentWithRequiredTools(MockMCPAgent):
            required_tools: ClassVar[list[str]] = ["required_tool"]

        tools = [types.Tool(name="required_tool", description="Required", inputSchema={})]
        ctx = MockEvalContext(prompt="Do something", tools=tools)
        agent = AgentWithRequiredTools()

        result = await agent.run(ctx)
        assert result.done


class TestMCPAgentOnToolsReady:
    """Tests for _on_tools_ready hook."""

    @pytest.mark.asyncio
    async def test_on_tools_ready_called(self) -> None:
        """Test _on_tools_ready is called during initialization."""
        hook_called = [False]

        class AgentWithHook(MockMCPAgent):
            def _on_tools_ready(self) -> None:
                hook_called[0] = True

        ctx = MockEvalContext(prompt="Do something")
        agent = AgentWithHook()

        await agent.run(ctx)
        assert hook_called[0]

    @pytest.mark.asyncio
    async def test_on_tools_ready_has_access_to_tools(self) -> None:
        """Test _on_tools_ready can access discovered tools."""
        captured_tools: list[types.Tool] = []

        class AgentWithHook(MockMCPAgent):
            def _on_tools_ready(self) -> None:
                captured_tools.extend(self.get_available_tools())

        tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={}),
            types.Tool(name="tool2", description="Tool 2", inputSchema={}),
        ]
        ctx = MockEvalContext(prompt="Do something", tools=tools)
        agent = AgentWithHook()

        await agent.run(ctx)

        assert len(captured_tools) == 2
        assert captured_tools[0].name == "tool1"


class TestMCPAgentToolSchemas:
    """Tests for tool schema generation."""

    @pytest.mark.asyncio
    async def test_get_tool_schemas(self) -> None:
        """Test get_tool_schemas returns correct format."""
        tools = [
            types.Tool(
                name="my_tool",
                description="My tool description",
                inputSchema={"type": "object", "properties": {"x": {"type": "string"}}},
            )
        ]
        ctx = MockEvalContext(prompt="Do something", tools=tools)
        agent = MockMCPAgent()

        # Initialize agent
        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        schemas = agent.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "my_tool"
        assert schemas[0]["description"] == "My tool description"


class TestMCPAgentErrorPropagation:
    """Tests for error propagation to EvalContext."""

    @pytest.mark.asyncio
    async def test_exception_propagates_to_ctx_error(self) -> None:
        """Test that exceptions during run() set ctx.error for platform visibility."""

        class FailingAgent(MockMCPAgent):
            async def get_response(self, messages: list[dict[str, Any]]) -> InferenceResult:
                raise RuntimeError("Agent crashed")

        ctx = MockEvalContext(prompt="Do something")
        agent = FailingAgent()

        result = await agent.run(ctx)

        # Should return error trace
        assert result.isError is True
        assert result.content is not None
        assert "Agent crashed" in result.content

        assert ctx.error is not None
        assert isinstance(ctx.error, BaseException)
        assert "Agent crashed" in str(ctx.error)

    @pytest.mark.asyncio
    async def test_step_error_propagates_to_ctx_error(self) -> None:
        """Test that step-level errors (caught internally) set ctx.error."""
        step_count = [0]

        class FailOnSecondStepAgent(MockMCPAgent):
            async def get_response(self, messages: list[dict[str, Any]]) -> InferenceResult:
                step_count[0] += 1
                if step_count[0] == 1:
                    return InferenceResult(
                        content="",
                        tool_calls=[MCPToolCall(name="test_tool", arguments={})],
                        done=False,
                    )
                else:
                    raise ValueError("Step 2 failed")

        ctx = MockEvalContext(prompt="Do something")
        agent = FailOnSecondStepAgent()

        result = await agent.run(ctx)

        # Should return error trace
        assert result.isError is True
        assert ctx.error is not None
        assert "Step 2 failed" in str(ctx.error)

    @pytest.mark.asyncio
    async def test_no_error_when_successful(self) -> None:
        """Test that ctx.error remains None on successful run."""
        ctx = MockEvalContext(prompt="Do something")
        agent = MockMCPAgent()

        result = await agent.run(ctx)

        assert result.isError is False
        assert ctx.error is None


class TestMCPAgentCategorizeTools:
    """Tests for the categorize_tools method."""

    @pytest.mark.asyncio
    async def test_categorize_generic_tools(self) -> None:
        """Test that tools without native specs are categorized as generic."""
        tools = [
            types.Tool(name="tool1", description="Tool 1", inputSchema={}),
            types.Tool(name="tool2", description="Tool 2", inputSchema={}),
        ]
        ctx = MockEvalContext(prompt="Test", tools=tools)
        agent = MockMCPAgent()
        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        categorized = agent.categorize_tools()

        assert len(categorized.generic) == 2
        assert len(categorized.native) == 0
        assert len(categorized.hosted) == 0
        assert len(categorized.skipped) == 0

    @pytest.mark.asyncio
    async def test_categorize_native_tools(self) -> None:
        """Test that tools with native specs are categorized correctly."""
        native_tool = types.Tool(
            name="native_tool",
            description="Native tool",
            inputSchema={},
            _meta={
                "native_tools": {
                    "integration_test": {
                        "api_type": "test_type",
                        "role": "test_role",
                    }
                }
            },
        )
        generic_tool = types.Tool(name="generic", description="Generic", inputSchema={})
        tools = [native_tool, generic_tool]

        ctx = MockEvalContext(prompt="Test", tools=tools)
        agent = MockMCPAgent()
        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        categorized = agent.categorize_tools()

        assert len(categorized.native) == 1
        assert categorized.native[0][0].name == "native_tool"
        assert len(categorized.generic) == 1
        assert categorized.generic[0].name == "generic"
        assert "test_role" in categorized.claimed_roles

    @pytest.mark.asyncio
    async def test_categorize_role_exclusion(self) -> None:
        """Test that tools with claimed roles are skipped."""
        # Native tool claims the "computer" role
        native_tool = types.Tool(
            name="claude_computer",
            description="Claude computer",
            inputSchema={},
            _meta={
                "native_tools": {
                    "integration_test": {
                        "api_type": "computer_test",
                        "role": "computer",
                    }
                }
            },
        )
        # Another computer tool that should be skipped
        other_computer = types.Tool(
            name="gemini_computer",
            description="Gemini computer",
            inputSchema={},
            _meta={
                "native_tools": {
                    "gemini": {
                        "api_type": "computer_use",
                        "role": "computer",
                    }
                }
            },
        )
        tools = [native_tool, other_computer]

        ctx = MockEvalContext(prompt="Test", tools=tools)
        agent = MockMCPAgent()
        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        categorized = agent.categorize_tools()

        assert len(categorized.native) == 1
        assert categorized.native[0][0].name == "claude_computer"
        assert len(categorized.skipped) == 1
        assert categorized.skipped[0][0].name == "gemini_computer"
        assert "computer" in categorized.claimed_roles

    @pytest.mark.asyncio
    async def test_categorize_hosted_tools(self) -> None:
        """Test that hosted tools are categorized separately."""
        hosted_tool = types.Tool(
            name="google_search",
            description="Google Search",
            inputSchema={},
            _meta={
                "native_tools": {
                    "integration_test": {
                        "api_type": "google_search",
                        "hosted": True,
                    }
                }
            },
        )
        tools = [hosted_tool]

        ctx = MockEvalContext(prompt="Test", tools=tools)
        agent = MockMCPAgent()
        agent.ctx = ctx
        await agent._initialize_from_ctx(ctx)

        categorized = agent.categorize_tools()

        assert len(categorized.hosted) == 1
        assert categorized.hosted[0][0].name == "google_search"
        assert categorized.hosted[0][1].hosted is True
        assert len(categorized.native) == 0
