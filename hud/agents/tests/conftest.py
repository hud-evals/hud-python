# pyright: reportPrivateUsage=false
"""Shared behavioral harness for agent tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast

import pytest
from mcp import types

from hud.agents.base import AgentState, MCPAgent
from hud.agents.tools import (
    AgentTool,
    AgentTools,
    AgentToolSpec,
    GroupedCapabilityMixin,
    ToolMetadata,
)
from hud.agents.tools.base import ToolClient
from hud.agents.types import AgentConfig
from hud.environment.router import ToolRouter
from hud.environment.scenarios import ScenarioSession
from hud.eval.context import EvalContext
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Trace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


class HarnessConfig(AgentConfig):
    model_name: str = "HarnessAgent"
    model: str = "harness-model"


def mcp_tool(name: str, *, description: str | None = None) -> types.Tool:
    return types.Tool(
        name=name,
        description=description or f"{name} tool",
        inputSchema={"type": "object", "properties": {}},
    )


def text_prompt(text: str, *, role: types.Role = "user") -> types.PromptMessage:
    return types.PromptMessage(
        role=role,
        content=types.TextContent(type="text", text=text),
    )


def text_result(text: str, *, is_error: bool = False) -> MCPToolResult:
    return MCPToolResult(
        content=[types.TextContent(type="text", text=text)],
        isError=is_error,
    )


def result_text(result: MCPToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))


class HarnessTool(AgentTool[dict[str, Any], dict[str, Any]]):
    name = "function"
    capability = "function"

    @classmethod
    def from_tool(cls, tool: types.Tool) -> HarnessTool:
        return cls(
            env_tool_name=tool.name,
            spec=AgentToolSpec(api_type="function", api_name=tool.name),
        )

    @property
    def provider_name(self) -> str:
        return self.env_tool_name

    def to_params(self) -> dict[str, Any]:
        return {"name": self.provider_name}

    def format_result(self, call: MCPToolCall, result: MCPToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "name": call.name,
            "content": result_text(result),
            "is_error": result.isError,
        }


class HarnessTools(AgentTools[HarnessTool, dict[str, Any], dict[str, Any]]):
    function_tool_class = HarnessTool


class HarnessNativeShellTool(HarnessTool):
    name = "shell"
    capability = "shell"

    @property
    def provider_name(self) -> str:
        return self.name

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return AgentToolSpec(api_type="shell", api_name="shell")


class HarnessFilesystemReadTool(GroupedCapabilityMixin, HarnessTool):
    name = "read_file"
    capability = "filesystem"
    env_tool_names: ClassVar[tuple[str, ...]] = ("read", "read_file")

    @property
    def provider_name(self) -> str:
        return self.name

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec:
        del model
        return AgentToolSpec(api_type="function", api_name="read_file")


class RoutingHarnessTools(AgentTools[HarnessTool, dict[str, Any], dict[str, Any]]):
    native_tool_classes = (HarnessNativeShellTool, HarnessFilesystemReadTool)
    function_tool_class = HarnessTool
    name_fallbacks: ClassVar[Mapping[str, tuple[str, ...]]] = {"shell": ("bash",)}


HarnessAgentTools: TypeAlias = AgentTools[HarnessTool, dict[str, Any], dict[str, Any]]


class HarnessAgentState(AgentState[dict[str, Any], HarnessAgentTools]):
    pass


class ScriptedAgent(MCPAgent[dict[str, Any], HarnessAgentTools, HarnessAgentState]):
    """Agent fake that exercises the real `MCPAgent.run` loop."""

    def __init__(
        self,
        responses: list[AgentResponse | BaseException],
        *,
        config: HarnessConfig | None = None,
        tools_factory: Callable[[], HarnessAgentTools] | None = None,
    ) -> None:
        super().__init__(config or HarnessConfig())
        self.config: HarnessConfig
        self.responses = list(responses)
        self.seen_messages: list[list[dict[str, Any]]] = []
        self._tools_factory = tools_factory or HarnessTools

    async def initialize_state(
        self,
        prompt: list[types.PromptMessage],
    ) -> HarnessAgentState:
        formatted: list[dict[str, Any]] = []
        for message in prompt:
            content = message.content
            formatted.append(
                {
                    "role": message.role,
                    "content": content.text if isinstance(content, types.TextContent) else "",
                }
            )
        return HarnessAgentState.model_construct(
            messages=formatted,
            tools=self._tools_factory(),
        )

    async def get_response(self, state: HarnessAgentState) -> AgentResponse:
        self.seen_messages.append([dict(message) for message in state.messages])
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class RecordingToolEnvironment:
    """Records the environment-facing MCP calls made by an agent run."""

    def __init__(
        self,
        tools: list[types.Tool] | None = None,
        *,
        results: Mapping[str, MCPToolResult | Exception] | None = None,
        tool_metadata: ToolMetadata | None = None,
    ) -> None:
        self.tools = tools or []
        self.results = dict(results or {})
        self.tool_metadata = tool_metadata
        self.calls: list[MCPToolCall] = []

    @property
    def client(self) -> ToolClient:
        return ToolClient(
            tools=self.tools,
            tool_handler=self.call_tool,
            tool_metadata=self.tool_metadata,
        )

    async def call_tool(self, call: MCPToolCall) -> MCPToolResult:
        self.calls.append(call)
        result = self.results.get(call.name, text_result(f"result from {call.name}"))
        if isinstance(result, Exception):
            raise result
        return result


class HarnessEvalContext(EvalContext):
    """Small EvalContext double that keeps the real `_run` and prompt behavior."""

    def __init__(
        self,
        prompt: str = "Test prompt",
        *,
        tools: list[types.Tool] | None = None,
        tool_results: Mapping[str, MCPToolResult | Exception] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.prompt = prompt
        self.environment = RecordingToolEnvironment(tools or [], results=tool_results)
        self._submitted: str | dict[str, Any] | None = None
        self.reward: float | None = None
        self._router = ToolRouter()
        self._scenario_sessions = {}
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
        self.metadata = metadata or {}
        self.results: list[Any] = []
        self._is_summary = False
        self._eval_api_key: str | None = None
        self._trace_enabled = False

    def as_tools(self) -> list[types.Tool]:
        return self.environment.tools

    @property
    def submitted(self) -> str | dict[str, Any] | None:
        return self._submitted

    def set_scenario_messages(self, messages: list[types.PromptMessage]) -> None:
        self._scenario_sessions["__client__"] = ScenarioSession(
            local_name="chat",
            full_name="test-env:chat",
            is_local=True,
            connection_name=None,
            resource_uri="test-env:chat",
            prompt_messages=messages,
        )

    def tool_metadata_for_run(self) -> ToolMetadata | None:
        return self._tool_metadata()

    async def run_agent(self, agent: Any, *, max_steps: int = 10) -> Trace:
        return await self._run(agent, max_steps=max_steps)

    async def list_tools(self, **kwargs: Any) -> list[types.Tool]:
        del kwargs
        return self.environment.tools

    async def call_tool(self, call: Any, /, **kwargs: Any) -> MCPToolResult:
        if isinstance(call, MCPToolCall):
            tool_call = call
        elif isinstance(call, tuple):
            call_tuple = cast("tuple[Any, ...]", call)
            tool_call = MCPToolCall(
                name=str(call_tuple[0]),
                arguments=cast("dict[str, Any]", call_tuple[1] if len(call_tuple) > 1 else {}),
            )
        else:
            tool_call = MCPToolCall(name=str(call), arguments=kwargs)
        return await self.environment.call_tool(tool_call)

    async def submit(self, answer: str | dict[str, Any]) -> None:
        self._submitted = answer


@pytest.fixture
def basic_tool() -> types.Tool:
    return mcp_tool("lookup")


@pytest.fixture
def recording_environment(basic_tool: types.Tool) -> RecordingToolEnvironment:
    return RecordingToolEnvironment([basic_tool])
