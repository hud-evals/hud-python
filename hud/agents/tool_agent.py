"""ToolAgent: catalog-driven provider tool-call loop.

Subclass contract::

    class ClaudeAgent(ToolAgent[BetaMessageParam]):
        tool_catalog = (ClaudeBashTool, ClaudeTextEditorTool, ClaudeMCPProxyTool)

        async def _initialize_state(self, *, prompt) -> RunState[BetaMessageParam]: ...
        async def get_response(self, state, *, system_prompt, citations_enabled): ...
        def _format_user_text(self, text) -> BetaMessageParam: ...
        def _format_result(self, call, result) -> BetaMessageParam | None: ...

``ToolAgent.run`` creates a fresh ``RunState`` per call and is fully re-entrant.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

import mcp.types as mcp_types

from hud.agents.base import Agent
from hud.agents.misc import auto_respond
from hud.capabilities import MCPClient
from hud.types import MCPToolCall, MCPToolResult, Trace

if TYPE_CHECKING:
    from hud.agents.tools.base import AgentTool
    from hud.agents.tools.hosted import HostedTool
    from hud.client import Manifest
    from hud.types import AgentResponse

logger = logging.getLogger(__name__)

MessageT = TypeVar("MessageT")


@dataclass
class ToolInvocation:
    """One tool call paired with its result."""

    call: MCPToolCall
    result: MCPToolResult


@dataclass
class RunState(Generic[MessageT]):
    """Mutable state for one agent run. Created fresh per ``run()`` call."""

    messages: list[MessageT] = field(default_factory=list)


class ToolAgent(Agent, Generic[MessageT]):
    """Catalog-driven provider tool-call loop."""

    tool_catalog: ClassVar[tuple[type[AgentTool[Any]], ...]] = ()

    # set by subclass __init__
    model: str
    auto_respond: bool
    hosted_tools: list[HostedTool[Any]]

    # populated by initialize
    tools: dict[str, AgentTool[Any]]
    params: list[Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "tool_catalog" in cls.__dict__:
            seen: dict[type, None] = {}
            for t in cls.tool_catalog:
                seen.setdefault(t.client_type, None)
            cls.clients = tuple(seen.keys())

    async def initialize(self, manifest: Manifest) -> None:
        await super().initialize(manifest)
        self.tools = {}
        self.params = []
        if not hasattr(self, "hosted_tools"):
            self.hosted_tools = []

        mcp_clients = [c for c in self.connections.values() if isinstance(c, MCPClient)]
        mcp_lists = await asyncio.gather(*(c.list_tools() for c in mcp_clients))
        mcp_by_client: dict[MCPClient, list[mcp_types.Tool]] = dict(
            zip(mcp_clients, mcp_lists, strict=False),
        )

        for tool_cls in type(self).tool_catalog:
            spec = tool_cls.default_spec(self.model)
            if spec is None:
                continue
            for client in self.connections.values():
                if not isinstance(client, tool_cls.client_type):
                    continue
                if isinstance(client, MCPClient):
                    for mt in mcp_by_client[client]:
                        tool = tool_cls(spec=spec, client=client, mcp_tool=mt)  # type: ignore[call-arg]
                        self.tools[tool.provider_name] = tool
                        self.params.append(tool.to_params())
                else:
                    tool = tool_cls(spec=spec, client=client)
                    self.tools[tool.provider_name] = tool
                    self.params.append(tool.to_params())

        for hosted in self.hosted_tools:
            if hosted.supports_model(self.model):
                self.params.append(hosted.to_params())

    async def run(
        self,
        *,
        prompt: str,
        max_steps: int = 10,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> Trace:
        try:
            state = await self._initialize_state(prompt=prompt)
            response: AgentResponse | None = None
            hit_max = False

            for step in range(1, max_steps + 1):
                logger.debug("step %d/%d", step, max_steps)
                response = await self.get_response(
                    state,
                    system_prompt=system_prompt,
                    citations_enabled=citations_enabled,
                )

                if response.done or not response.tool_calls:
                    follow_up = await auto_respond(response.content, enabled=self.auto_respond)
                    if follow_up is not None:
                        text = (
                            follow_up.content.text
                            if isinstance(follow_up.content, mcp_types.TextContent)
                            else ""
                        )
                        state.messages.append(self._format_user_text(text))
                        continue
                    break

                for call in response.tool_calls:
                    result = await self._dispatch_call(call)
                    msg = self._format_result(call, result)
                    if msg is None:
                        continue
                    if isinstance(msg, list):
                        state.messages.extend(cast("list[MessageT]", msg))
                    else:
                        state.messages.append(cast("MessageT", msg))

                if step == max_steps:
                    hit_max = True

            error: str | None = "max_steps_exceeded" if hit_max else None
            return Trace(
                done=True,
                messages=state.messages,
                content=response.content if response else (error or ""),
                isError=bool(error) or (response.isError if response else False),
                citations=(response.citations if response else None) or [],
                info={"error": error} if error else {},
            )
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.exception("ToolAgent.run failed")
            return Trace(done=True, content=str(exc), isError=True, info={"error": str(exc)})

    async def _dispatch_call(self, call: MCPToolCall) -> MCPToolResult:
        tool = self.tools.get(call.name)
        if tool is None:
            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text=f"unknown tool: {call.name!r}")],
                isError=True,
            )
        args = call.arguments if isinstance(call.arguments, dict) else {}
        try:
            return await tool.execute(args)
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as exc:
            logger.exception("tool %s failed", call.name)
            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text=f"tool error: {exc}")],
                isError=True,
            )

    # ─── provider hooks ───────────────────────────────────────────────

    @abstractmethod
    async def _initialize_state(self, *, prompt: str) -> RunState[MessageT]:
        """Build fresh run state from the prompt."""

    @abstractmethod
    async def get_response(
        self,
        state: RunState[MessageT],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentResponse:
        """Call the provider API with state.messages + self.params."""

    @abstractmethod
    def _format_user_text(self, text: str) -> MessageT:
        """Wrap a plain text string as a provider user message."""

    @abstractmethod
    def _format_result(
        self, call: MCPToolCall, result: MCPToolResult,
    ) -> MessageT | list[MessageT] | None:
        """Convert a tool result into one or more provider messages, or None to skip."""


__all__ = ["RunState", "ToolAgent", "ToolInvocation"]
