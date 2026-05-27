"""Shared support for agent-owned harness tools."""

from __future__ import annotations

import fnmatch
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar, cast

import mcp.types as types

from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from hud.agents.tools.hosted import HostedTool

AgentToolParamT_co = TypeVar("AgentToolParamT_co", covariant=True)
MessageT_co = TypeVar("MessageT_co", covariant=True)
ToolParamT = TypeVar("ToolParamT")
MessageT = TypeVar("MessageT")
AgentToolT = TypeVar("AgentToolT", bound="AgentTool[Any, Any]")
CallTool = Callable[[MCPToolCall], Awaitable[MCPToolResult]]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolClient:
    """MCP tools and execution hook available for one agent run."""

    tools: list[types.Tool] = field(default_factory=list[types.Tool])
    tool_handler: CallTool | None = None


@dataclass(frozen=True)
class AgentToolSpec:
    """Provider tool definition owned by an agent harness."""

    api_type: str
    api_name: str
    supported_models: tuple[str, ...] | None = None

    def supports_model(self, model: str | None) -> bool:
        if not self.supported_models:
            return True
        if not model or model == "unknown":
            return False
        model_lower = model.lower()
        return any(
            fnmatch.fnmatch(model_lower, pattern.lower()) for pattern in self.supported_models
        )


class AgentTool(ABC, Generic[AgentToolParamT_co, MessageT_co]):
    """Provider-facing tool owned by an agent harness."""

    name: ClassVar[str]
    capability: ClassVar[str]

    def __init__(self, *, env_tool_name: str, spec: AgentToolSpec) -> None:
        self.env_tool_name = env_tool_name
        self.spec = spec

    @property
    def provider_name(self) -> str:
        return self.name

    @classmethod
    def from_native_tool(
        cls,
        tool: types.Tool,
        model: str,
    ) -> Self | None:
        spec = cls.default_spec(model)
        if spec is None:
            return None
        return cls(env_tool_name=tool.name, spec=spec)

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec | None:
        """Return the provider spec this agent should use for this capability."""
        return None

    @classmethod
    def from_tool(cls, tool: types.Tool) -> Self | None:
        """Build a provider tool for a generic environment tool."""
        del tool
        return None

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute an environment-backed tool by forwarding to its MCP tool."""
        return await call_tool(MCPToolCall(name=self.env_tool_name, arguments=arguments))

    def format_result(
        self, call: MCPToolCall, result: MCPToolResult
    ) -> MessageT_co | list[MessageT_co] | None:
        """Format a single tool result for the provider continuation turn."""
        del result
        logger.warning("Tool '%s' does not implement result formatting.", call.name)
        return None

    @abstractmethod
    def to_params(self) -> AgentToolParamT_co: ...


class AgentTools(dict[str, AgentToolT], Generic[AgentToolT, ToolParamT, MessageT]):
    """Prepared tool state owned by a single agent run."""

    native_tool_classes: ClassVar[tuple[type[AgentTool[Any, Any]], ...]] = ()
    function_tool_class: ClassVar[type[AgentTool[Any, Any]] | None] = None

    def __init__(self) -> None:
        super().__init__()
        self.params: list[ToolParamT] = []
        self.hosted_tools: list[HostedTool[object]] = []

    def select_tools(
        self,
        tools: list[types.Tool],
        model: str,
    ) -> tuple[list[AgentToolT], list[types.Tool]]:
        """Split MCP tools into provider-owned and user-defined tools."""
        logger.info("Discovered %s tools: %s", len(tools), ", ".join(tool.name for tool in tools))

        tools_by_capability: dict[str, types.Tool] = {}
        for tool in tools:
            meta = tool.meta
            capability = meta.get("capability") if isinstance(meta, dict) else None
            if isinstance(capability, str) and capability:
                tools_by_capability[capability] = tool

        agent_tools: list[AgentToolT] = []
        for raw_tool_cls in self.native_tool_classes:
            tool_cls = cast("type[AgentToolT]", raw_tool_cls)
            native_tool = tools_by_capability.get(tool_cls.capability)
            if native_tool is None:
                continue
            agent_tool = tool_cls.from_native_tool(native_tool, model)
            if agent_tool is not None:
                agent_tools.append(agent_tool)
        agent_tool_names = {tool.env_tool_name for tool in agent_tools}
        user_tools = [tool for tool in tools if tool.name not in agent_tool_names]
        return agent_tools, user_tools

    def generic_tool(
        self,
        tool: types.Tool,
    ) -> ToolParamT | None:
        """Convert an environment MCP tool into provider params."""
        del tool
        return None

    def prepare(
        self,
        *,
        model: str,
        tools: list[types.Tool],
        hosted_tools: list[HostedTool[object]] | None = None,
    ) -> None:
        """Prepare a generic provider tool map for an agent run."""
        self.clear()
        self.params = []
        self.hosted_tools = []

        provider_tools, user_tools = self.select_tools(
            tools,
            model,
        )
        tools_by_name = {tool.provider_name: tool for tool in provider_tools}
        installed_names = set(tools_by_name)
        self.update(tools_by_name)
        self.params.extend(cast("ToolParamT", tool.to_params()) for tool in provider_tools)

        selected_hosted_tools: list[HostedTool[object]] = []
        for tool in hosted_tools or []:
            if not tool.supports_model(model):
                continue
            selected_hosted_tools.append(tool)
            self.params.append(cast("ToolParamT", tool.to_params()))
        self.hosted_tools = selected_hosted_tools

        for tool in user_tools:
            if self.function_tool_class is not None:
                function_tool_cls = cast("type[AgentToolT]", self.function_tool_class)
                agent_tool = function_tool_cls.from_tool(tool)
                if agent_tool is None:
                    continue
                self[agent_tool.provider_name] = agent_tool
                installed_names.add(agent_tool.provider_name)
                self.params.append(cast("ToolParamT", agent_tool.to_params()))
                continue
            generic_tool = self.generic_tool(tool)
            if generic_tool is None:
                continue
            installed_names.add(tool.name)
            self.params.append(generic_tool)

        tool_names = sorted(installed_names)
        logger.info("Agent initialized with %s tools: %s", len(tool_names), ", ".join(tool_names))

    async def execute(
        self,
        call_tool: CallTool | None,
        tool_call: MCPToolCall | list[MCPToolCall] | None = None,
    ) -> list[MessageT]:
        if tool_call is None:
            return []

        if call_tool is None:
            raise ValueError("call_tool callback is required to execute tool calls")

        outputs: list[MessageT] = []
        tool_calls = [tool_call] if isinstance(tool_call, MCPToolCall) else tool_call
        for tc in tool_calls:
            agent_tool = self[tc.name]
            arguments = tc.arguments if isinstance(tc.arguments, dict) else {}
            try:
                result = await agent_tool.execute(call_tool, arguments)
            except TimeoutError:
                raise
            except Exception as exc:
                logger.exception("Tool execution failed")
                result = MCPToolResult(
                    content=[types.TextContent(type="text", text=str(exc))],
                    isError=True,
                )

            output = cast("MessageT | list[MessageT] | None", agent_tool.format_result(tc, result))
            if output is None:
                continue
            if isinstance(output, list):
                outputs.extend(cast("list[MessageT]", output))
            else:
                outputs.append(output)

        return outputs
