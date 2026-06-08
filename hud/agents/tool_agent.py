"""ToolAgent: provider-neutral tool-call loop.

Subclass contract::

    class ClaudeAgent(ToolAgent[BetaMessageParam, BetaToolUnionParam, ClaudeConfig]):
        tools = (ClaudeBashTool, ClaudeTextEditorTool, ClaudeComputerTool, ClaudeMCPProxyTool)

        async def _initialize_state(self, *, prompt) -> RunState[...]: ...
        async def get_response(self, state, *, system_prompt, citations_enabled): ...
        def _format_message(self, role, text) -> BetaMessageParam: ...
        def _format_result(self, call, result) -> BetaMessageParam | None: ...

``RunState`` carries the messages *and* the tools/params built for one run, so a
single agent instance can drive many concurrent ``rollout`` calls with no shared
mutable state. It is generic over both the provider message type (``MessageT``)
and the provider tool-param type (``ParamT``), so ``state.params`` is the
provider's own tool-param list rather than ``list[Any]``.
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
from hud.telemetry.instrument import instrument
from hud.types import MCPToolCall, MCPToolResult

if TYPE_CHECKING:
    from hud.agents.tools.base import AgentTool
    from hud.agents.tools.hosted import HostedTool
    from hud.agents.types import ToolAgentConfig
    from hud.capabilities import CapabilityClient
    from hud.client import Run
    from hud.types import AgentResponse

logger = logging.getLogger(__name__)

MessageT = TypeVar("MessageT")
ParamT = TypeVar("ParamT")
ConfigT = TypeVar("ConfigT", bound="ToolAgentConfig")


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
class RunState(Generic[MessageT, ParamT]):
    """Mutable per-run state: messages + the tools/params built for this run.

    Created fresh per ``rollout`` (or ``run``) call, so one agent instance can
    drive many concurrent rollouts without shared mutable state.
    """

    messages: list[MessageT] = field(default_factory=lambda: cast("list[MessageT]", []))
    tools: dict[str, AgentTool[Any]] = field(
        default_factory=lambda: cast("dict[str, AgentTool[Any]]", {}),
    )
    params: list[ParamT] = field(default_factory=lambda: cast("list[ParamT]", []))


class ToolAgent(Agent, Generic[MessageT, ParamT, ConfigT]):
    """Provider-neutral tool-call loop."""

    #: Provider-facing tool classes this agent can advertise and execute.
    tools: ClassVar[tuple[type[AgentTool[Any]], ...]] = ()

    #: Per-agent configuration, parametrized via ``ConfigT``; each subclass
    #: __init__ sets this and the loop reads everything it needs through it.
    config: ConfigT

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def auto_respond(self) -> bool:
        return self.config.auto_respond

    @property
    def hosted_tools(self) -> list[HostedTool[Any]]:
        return self.config.hosted_tools

    async def __call__(
        self,
        run: Run,
        *,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> None:
        """Drive this (stateless) agent over a live ``Run``, filling ``run.trace``.

        Opens the capabilities this agent's tools require off the connection
        (``run.client.open(protocol)``), builds the tools into a fresh ``RunState``,
        then runs the loop against ``run.prompt``, accumulating the trajectory onto
        ``run.trace``. No per-rollout state is stored on ``self``, so one instance
        may drive many concurrent rollouts.
        """
        connections: dict[str, CapabilityClient] = {}
        manifest = run.client.manifest
        if manifest is not None:
            wanted = {tool_cls.client_type.protocol for tool_cls in type(self).tools}
            for cap in manifest.bindings:
                if cap.protocol in wanted and cap.protocol not in connections:
                    connections[cap.protocol] = await run.client.open(cap.protocol)
        state = await self._initialize_state(prompt=run.prompt)
        state.tools, state.params = await self._build_tools(connections)
        await self._loop(
            run,
            state,
            max_steps=10 if max_steps is None else max_steps,
            system_prompt=system_prompt,
            citations_enabled=citations_enabled,
        )

    async def _build_tools(
        self,
        connections: dict[str, CapabilityClient],
    ) -> tuple[dict[str, AgentTool[Any]], list[ParamT]]:
        """Build executable tools and provider API params for one run."""
        tools: dict[str, AgentTool[Any]] = {}
        params: list[ParamT] = []
        for tool_cls in type(self).tools:
            bound_tools, bound_params = await tool_cls.bind(
                model=self.model,
                connections=connections,
            )
            tools.update(bound_tools)
            params.extend(cast("list[ParamT]", bound_params))
        params.extend(
            cast("ParamT", hosted.to_params())
            for hosted in self.hosted_tools
            if hosted.supports_model(self.model)
        )
        return tools, params

    async def _loop(
        self,
        run: Run,
        state: RunState[MessageT, ParamT],
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

                if response.isError:
                    break

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
            trace.finish(
                response.content if response else (error or ""),
                isError=bool(error) or (response.isError if response else False),
                error=error,
                citations=(response.citations if response else None) or [],
                messages=state.messages,
            )
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.exception("ToolAgent loop failed")
            trace.fail(str(exc))

    async def _dispatch_call(
        self,
        call: MCPToolCall,
        state: RunState[MessageT, ParamT],
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
    async def _initialize_state(
        self, *, prompt: str | list[Any] | None
    ) -> RunState[MessageT, ParamT]:
        """Build fresh run state from the prompt (use ``self._initial_messages``)."""

    @abstractmethod
    async def get_response(
        self,
        state: RunState[MessageT, ParamT],
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
        state: RunState[MessageT, ParamT],
    ) -> MessageT | list[MessageT] | None:
        """Convert a tool result into one or more provider messages, or None to skip."""


__all__ = ["RunState", "ToolAgent", "to_prompt_messages"]
