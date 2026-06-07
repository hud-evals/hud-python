"""ToolAgent: catalog-driven provider tool-call loop.

Subclass contract::

    class ClaudeAgent(ToolAgent[BetaMessageParam]):
        tool_catalog = (ClaudeBashTool, ClaudeTextEditorTool, ClaudeMCPProxyTool)

        async def _initialize_state(self, *, prompt) -> RunState[BetaMessageParam]: ...
        async def get_response(self, state, *, system_prompt, citations_enabled): ...
        def _format_message(self, role, text) -> BetaMessageParam: ...
        def _format_result(self, call, result) -> BetaMessageParam | None: ...

``RunState`` carries the messages *and* the tools/params built for one run, so a
single agent instance can drive many concurrent ``rollout`` calls with no shared
mutable state.
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
from hud.telemetry.instrument import instrument
from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from hud.agents.tools.base import AgentTool
    from hud.agents.tools.hosted import HostedTool
    from hud.capabilities import CapabilityClient
    from hud.client import Run
    from hud.types import AgentResponse

logger = logging.getLogger(__name__)

MessageT = TypeVar("MessageT")


def _message_text(message: mcp_types.PromptMessage) -> str:
    """Best-effort plain text for a prompt message (text content only for now)."""
    content = message.content
    if isinstance(content, mcp_types.TextContent):
        return content.text
    return getattr(content, "text", "") or ""


def to_prompt_messages(prompt: str | list[Any] | None) -> list[mcp_types.PromptMessage]:
    """Normalize a task prompt into a list of ``PromptMessage`` turns.

    Accepts the two shapes a ``Run.prompt`` can take: plain text (one user turn)
    or a list of message dicts / ``PromptMessage`` objects (chat-style, multi-turn).
    """
    if prompt is None:
        prompt = ""
    if isinstance(prompt, str):
        return [
            mcp_types.PromptMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text=prompt),
            ),
        ]
    messages: list[mcp_types.PromptMessage] = []
    for item in prompt:
        if isinstance(item, mcp_types.PromptMessage):
            messages.append(item)
        elif isinstance(item, dict):
            messages.append(mcp_types.PromptMessage.model_validate(item))
        else:
            messages.append(
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(type="text", text=str(item)),
                ),
            )
    return messages


@dataclass
class RunState(Generic[MessageT]):
    """Mutable per-run state: messages + the tools/params built for this run.

    Created fresh per ``rollout`` (or ``run``) call, so one agent instance can
    drive many concurrent rollouts without shared mutable state.
    """

    messages: list[MessageT] = field(default_factory=list)
    tools: dict[str, AgentTool[Any]] = field(default_factory=dict)
    params: list[Any] = field(default_factory=list)


class ToolAgent(Agent, Generic[MessageT]):
    """Catalog-driven provider tool-call loop."""

    tool_catalog: ClassVar[tuple[type[AgentTool[Any]], ...]] = ()
    #: Capability-client types this agent can drive (derived from the catalog).
    clients: ClassVar[tuple[type[CapabilityClient], ...]] = ()

    # set by subclass __init__
    model: str
    auto_respond: bool
    hosted_tools: list[HostedTool[Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "tool_catalog" in cls.__dict__:
            seen: dict[type, None] = {}
            for t in cls.tool_catalog:
                seen.setdefault(t.client_type, None)
            cls.clients = tuple(seen.keys())

    async def __call__(
        self,
        run: Run,
        *,
        max_steps: int = 10,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> None:
        """Drive this (stateless) agent over a live ``Run``, filling ``run.trace``.

        Opens the capabilities this agent's catalog supports off the connection
        (``run.client.open(protocol)``), builds the tools into a fresh ``RunState``,
        then runs the loop against ``run.prompt``, accumulating the trajectory onto
        ``run.trace``. No per-rollout state is stored on ``self``, so one instance
        may drive many concurrent rollouts.
        """
        connections: dict[str, CapabilityClient] = {}
        manifest = run.client.manifest
        if manifest is not None:
            wanted = {cls.protocol for cls in type(self).clients}
            for cap in manifest.bindings:
                if cap.protocol in wanted and cap.protocol not in connections:
                    connections[cap.protocol] = await run.client.open(cap.protocol)
        state = await self._initialize_state(prompt=run.prompt)
        state.tools, state.params = await self._build_tools(connections)
        await self._loop(
            run,
            state,
            max_steps=max_steps,
            system_prompt=system_prompt,
            citations_enabled=citations_enabled,
        )

    async def _build_tools(
        self,
        connections: dict[str, CapabilityClient],
    ) -> tuple[dict[str, AgentTool[Any]], list[Any]]:
        """Build the (tools, params) for one run from the given open connections."""
        tools: dict[str, AgentTool[Any]] = {}
        params: list[Any] = []
        hosted_tools = getattr(self, "hosted_tools", [])

        mcp_clients = [c for c in connections.values() if isinstance(c, MCPClient)]
        mcp_lists = await asyncio.gather(*(c.list_tools() for c in mcp_clients))
        mcp_by_client: dict[MCPClient, list[mcp_types.Tool]] = dict(
            zip(mcp_clients, mcp_lists, strict=False),
        )

        for tool_cls in type(self).tool_catalog:
            spec = tool_cls.default_spec(self.model)
            if spec is None:
                continue
            for client in connections.values():
                if not isinstance(client, tool_cls.client_type):
                    continue
                if isinstance(client, MCPClient):
                    for mt in mcp_by_client[client]:
                        tool = tool_cls(spec=spec, client=client, mcp_tool=mt)  # type: ignore[call-arg]
                        tools[tool.provider_name] = tool
                        params.append(tool.to_params())
                else:
                    tool = tool_cls(spec=spec, client=client)
                    tools[tool.provider_name] = tool
                    params.append(tool.to_params())

        params.extend(
            hosted.to_params() for hosted in hosted_tools if hosted.supports_model(self.model)
        )

        return tools, params

    async def _loop(
        self,
        run: Run,
        state: RunState[MessageT],
        *,
        max_steps: int = 10,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> None:
        trace = run.trace
        try:
            response: AgentResponse | None = None
            hit_max = False

            for step in range(1, max_steps + 1):
                logger.debug("step %d/%d", step, max_steps)
                response = await instrument(
                    self.get_response,
                    category="inference-2",
                    record_args=False,
                )(
                    state,
                    system_prompt=system_prompt,
                    citations_enabled=citations_enabled,
                )
                if response.sample is not None:
                    trace.samples.append(response.sample)

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
                    result = await self._dispatch_call(call, state)
                    msg = self._format_result(call, result, state)
                    if msg is None:
                        continue
                    if isinstance(msg, list):
                        state.messages.extend(cast("list[MessageT]", msg))
                    else:
                        state.messages.append(cast("MessageT", msg))

                if step == max_steps:
                    hit_max = True

            error: str | None = "max_steps_exceeded" if hit_max else None
            trace.done = True
            trace.messages = state.messages
            trace.content = response.content if response else (error or "")
            trace.isError = bool(error) or (response.isError if response else False)
            trace.citations = (response.citations if response else None) or []
            if error:
                trace.info["error"] = error
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.exception("ToolAgent loop failed")
            trace.done = True
            trace.content = str(exc)
            trace.isError = True
            trace.info["error"] = str(exc)

    async def _dispatch_call(
        self,
        call: MCPToolCall,
        state: RunState[MessageT],
    ) -> MCPToolResult:
        tool = state.tools.get(call.name)
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

    def _initial_messages(self, prompt: str | list[Any] | None) -> list[MessageT]:
        """Turn a run prompt (text or message list) into provider messages."""
        return [
            self._format_message(message.role, _message_text(message))
            for message in to_prompt_messages(prompt)
        ]

    @abstractmethod
    async def _initialize_state(self, *, prompt: str | list[Any] | None) -> RunState[MessageT]:
        """Build fresh run state from the prompt (use ``self._initial_messages``)."""

    @abstractmethod
    async def get_response(
        self,
        state: RunState[MessageT],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentResponse:
        """Call the provider API with ``state.messages`` + ``state.params``."""

    def _format_user_text(self, text: str) -> MessageT:
        """Wrap a plain text string as a provider user message."""
        return self._format_message("user", text)

    @abstractmethod
    def _format_message(self, role: str, text: str) -> MessageT:
        """Wrap text as a provider message of the given role (``user``/``assistant``)."""

    @abstractmethod
    def _format_result(
        self,
        call: MCPToolCall,
        result: MCPToolResult,
        state: RunState[MessageT],
    ) -> MessageT | list[MessageT] | None:
        """Convert a tool result into one or more provider messages, or None to skip."""


__all__ = ["RunState", "ToolAgent", "to_prompt_messages"]
