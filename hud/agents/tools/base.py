"""Shared support for agent-owned harness tools."""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar

from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from hud.agents.base import MCPAgent
    from hud.agents.tools.capabilities import EnvironmentCapability

ToolParamT = TypeVar("ToolParamT")
CallTool = Callable[[MCPToolCall], Awaitable[MCPToolResult]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Provider tool definition owned by an agent harness."""

    api_type: str
    api_name: str
    beta: str | None = None
    supported_models: tuple[str, ...] | None = None

    def supports_model(self, model: str | None) -> bool:
        if not self.supported_models or not model or model == "unknown":
            return True
        model_lower = model.lower()
        return any(
            fnmatch.fnmatch(model_lower, pattern.lower()) for pattern in self.supported_models
        )


class AgentTool(ABC, Generic[ToolParamT]):
    """Provider-facing tool backed by one environment tool."""

    name: ClassVar[str]
    capability: ClassVar[str]

    def __init__(self, *, env_tool_name: str, spec: AgentToolSpec) -> None:
        self.env_tool_name = env_tool_name
        self.spec = spec

    @classmethod
    def from_capability(
        cls,
        capability: EnvironmentCapability,
        spec: AgentToolSpec,
        model: str,
    ) -> Self:
        del model
        return cls(env_tool_name=capability.tool_name, spec=spec)

    @classmethod
    def default_spec(cls, model: str) -> AgentToolSpec | None:
        """Return the provider spec this agent should use for this capability."""
        del model
        return None

    @property
    def required_beta(self) -> str | None:
        return self.spec.beta

    async def execute(self, caller: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        """Execute by forwarding to the backing environment tool."""
        return await call_tool(caller, self.env_tool_name, arguments)

    @abstractmethod
    def to_params(self) -> ToolParamT: ...


async def call_tool(
    caller: CallTool,
    env_tool_name: str,
    arguments: dict[str, Any],
) -> MCPToolResult:
    result = await caller(MCPToolCall(name=env_tool_name, arguments=arguments))
    return MCPToolResult(content=result.content, isError=result.isError)


async def call_agent_tools(
    agent: MCPAgent,
    agent_tools: Mapping[str, AgentTool[Any]],
    tool_call: MCPToolCall | list[MCPToolCall] | None = None,
) -> list[MCPToolResult]:
    """Route provider-owned tool calls through adapters, otherwise through MCP."""
    import mcp.types as types

    from hud.agents.base import MCPAgent

    if tool_call is None:
        return []
    tool_calls = [tool_call] if isinstance(tool_call, MCPToolCall) else tool_call

    async def call_env_tool(call: MCPToolCall) -> MCPToolResult:
        return (await MCPAgent.call_tools(agent, call))[0]

    results: list[MCPToolResult] = []
    for tc in tool_calls:
        agent_tool = agent_tools.get(tc.name)
        if agent_tool is None:
            results.extend(await MCPAgent.call_tools(agent, tc))
            continue

        try:
            arguments = tc.arguments if isinstance(tc.arguments, dict) else {}
            results.append(await agent_tool.execute(call_env_tool, arguments))
        except Exception as exc:
            agent.console.error_log(f"Agent tool execution failed: {exc}")
            results.append(
                MCPToolResult(
                    content=[types.TextContent(type="text", text=str(exc))],
                    isError=True,
                )
            )
    return results


__all__ = ["AgentTool", "AgentToolSpec", "CallTool", "call_agent_tools", "call_tool"]
