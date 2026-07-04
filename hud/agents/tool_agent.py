"""ToolAgent: catalog-driven provider tool-call loop.

Subclass contract::

    class ClaudeAgent(ToolAgent[BetaMessageParam, ClaudeConfig]):
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
import json
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

import mcp.types as mcp_types

from hud.agents.base import Agent
from hud.agents.misc import auto_respond
from hud.agents.tools.base import AgentTool
from hud.agents.types import AgentStep, ToolStep
from hud.capabilities import MCPClient
from hud.types import AgentType, MCPToolCall, MCPToolResult, Step
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.agents.types import AgentConfig
    from hud.capabilities import CapabilityClient
    from hud.eval.run import Run

logger = logging.getLogger(__name__)

# Tool calls whose arguments never parsed as JSON (usually a completion truncated at the token
# limit): _dispatch_call answers the call id with an error result so the model can re-issue.
MALFORMED_TOOL_ARGS_KEY = "__hud_malformed_tool_arguments__"


def parse_tool_arguments(raw: str | None, *, tool_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        logger.warning("Malformed tool-call arguments for %s: %s", tool_name, e)
        return {MALFORMED_TOOL_ARGS_KEY: f"{e} in arguments: {(raw or '')[:200]!r}"}
    return cast("dict[str, Any]", parsed) if isinstance(parsed, dict) else {}


MessageT = TypeVar("MessageT")
ConfigT = TypeVar("ConfigT", bound="AgentConfig")


def _message_text(message: mcp_types.PromptMessage) -> str:
    """Best-effort plain text for a prompt message (text content only for now)."""
    content = message.content
    if isinstance(content, mcp_types.TextContent):
        return content.text
    return getattr(content, "text", "") or ""


@dataclass
class RunState(Generic[MessageT]):
    """Mutable per-run state: messages + the tools/params built for this run.

    Created fresh per ``rollout`` (or ``run``) call, so one agent instance can
    drive many concurrent rollouts without shared mutable state.
    """

    messages: list[MessageT] = field(default_factory=list[MessageT])
    tools: dict[str, AgentTool[Any]] = field(default_factory=dict[str, AgentTool[Any]])
    params: list[Any] = field(default_factory=list[Any])


class ToolAgent(Agent, Generic[MessageT, ConfigT]):
    """Catalog-driven provider tool-call loop."""

    tool_catalog: ClassVar[tuple[type[AgentTool[Any]], ...]] = ()
    #: Capability-client types this agent can drive (derived from the catalog).
    clients: ClassVar[tuple[type[CapabilityClient], ...]] = ()

    #: The agent's typed config; set by subclass __init__.
    config: ConfigT

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "tool_catalog" in cls.__dict__:
            seen: dict[type, None] = {}
            for t in cls.tool_catalog:
                seen.setdefault(t.client_type, None)
            cls.clients = tuple(seen.keys())

    def hosted_spec(self) -> dict[str, Any]:
        """HUD-hosted execution runs the agent remotely, so it is
        reconstructed there from this identity (type, model, step budget, system
        prompt) with the model resolved through the HUD gateway.
        """
        if self.config.model_client is not None:
            raise ValueError(
                "hosted execution cannot serialize a custom model_client; "
                "set the model by name and let the hosted runner build the gateway client"
            )
        agent_type = AgentType.of(self)
        if agent_type is None:
            raise ValueError(
                f"hosted execution supports the gateway agent types "
                f"({', '.join(at.value for at in AgentType)}); got {type(self).__name__}"
            )
        config = self.config.model_dump(
            mode="json",
            exclude={"model_client", "api_key", "base_url", "hosted_tools"},
        )
        return {"type": agent_type.value, "config": config}

    async def __call__(self, run: Run) -> None:
        """Drive this (stateless) agent over a live ``Run``, filling ``run.trace``.

        Opens the capabilities this agent's catalog supports off the connection
        (``run.client.open(protocol)``), builds the tools into a fresh ``RunState``,
        then runs the loop against ``run.prompt_messages``, accumulating the
        trajectory onto ``run.trace``. Loop budget and prompting come from the agent's config
        (``max_steps``, ``system_prompt``, ``citations_enabled``). No per-rollout
        state is stored on ``self``, so one instance may drive many concurrent
        rollouts.
        """
        connections: dict[str, CapabilityClient] = {}
        manifest = run.client.manifest
        if manifest is not None:
            wanted = {cls.protocol for cls in type(self).clients}
            for cap in manifest.bindings:
                if cap.protocol in wanted and cap.protocol not in connections:
                    connections[cap.protocol] = await run.client.open(cap.protocol)
        state = await self._initialize_state(prompt=run.prompt_messages)
        state.tools, state.params = await self._build_tools(connections)
        await self._loop(
            run,
            state,
            max_steps=self.config.max_steps,
            system_prompt=self.config.system_prompt,
            citations_enabled=self.config.citations_enabled,
        )

    async def _build_tools(
        self,
        connections: dict[str, CapabilityClient],
    ) -> tuple[dict[str, AgentTool[Any]], list[Any]]:
        """Build the (tools, params) for one run from the given open connections."""
        tools: dict[str, AgentTool[Any]] = {}
        params: list[Any] = []
        model = self.config.model
        hosted_tools = self.config.hosted_tools

        mcp_clients = [c for c in connections.values() if isinstance(c, MCPClient)]
        mcp_lists = await asyncio.gather(*(c.list_tools() for c in mcp_clients))
        mcp_by_client: dict[MCPClient, list[mcp_types.Tool]] = dict(
            zip(mcp_clients, mcp_lists, strict=False),
        )

        for tool_cls in type(self).tool_catalog:
            spec = tool_cls.default_spec(model)
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

        params.extend(hosted.to_params() for hosted in hosted_tools if hosted.supports_model(model))

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
            step: AgentStep | None = None
            hit_max = False

            for turn in range(1, max_steps + 1):
                logger.info("step %d/%d", turn, max_steps)
                started_at = now_iso()
                step = await self.get_response(
                    state,
                    system_prompt=system_prompt,
                    citations_enabled=citations_enabled,
                )
                step.started_at = step.started_at or started_at
                step.model = step.model or self.config.model
                run.record(step)

                if step.tool_calls:
                    logger.info("  → %s", ", ".join(c.name for c in step.tool_calls))

                if step.done or not step.tool_calls:
                    follow_up = await auto_respond(step.content, enabled=self.config.auto_respond)
                    if follow_up is not None:
                        text = (
                            follow_up.content.text
                            if isinstance(follow_up.content, mcp_types.TextContent)
                            else ""
                        )
                        state.messages.append(self._format_user_text(text))
                        run.record(Step(source="user", messages=[follow_up]))
                        continue
                    break

                for call in step.tool_calls:
                    call_started_at = now_iso()
                    result = await self._dispatch_call(call, state)
                    run.record(ToolStep(call=call, result=result, started_at=call_started_at))
                    msg = self._format_result(call, result, state)
                    if msg is None:
                        continue
                    if isinstance(msg, list):
                        state.messages.extend(cast("list[MessageT]", msg))
                    else:
                        state.messages.append(cast("MessageT", msg))

                if turn == max_steps:
                    hit_max = True

            trace.content = step.content if step else None
            trace.status = "error" if step is not None and step.error else "completed"
            trace.extra["stop_reason"] = "max_steps" if hit_max else "done"
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.exception("ToolAgent loop failed")
            trace.status = "error"
            run.record(Step(source="system", error=str(exc)))

    async def _dispatch_call(
        self,
        call: MCPToolCall,
        state: RunState[MessageT],
    ) -> MCPToolResult:
        args_maybe = call.arguments if isinstance(call.arguments, dict) else {}
        if MALFORMED_TOOL_ARGS_KEY in args_maybe:
            return MCPToolResult(
                content=[
                    mcp_types.TextContent(
                        type="text",
                        text=(
                            f"tool error: arguments for {call.name!r} were not valid JSON "
                            f"(response likely truncated at the output-token limit): "
                            f"{args_maybe[MALFORMED_TOOL_ARGS_KEY]}. "
                            f"Re-issue the tool call with complete JSON arguments."
                        ),
                    )
                ],
                isError=True,
            )
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

    def _initial_messages(self, prompt: list[mcp_types.PromptMessage]) -> list[MessageT]:
        """Map normalized prompt turns onto provider messages."""
        return [self._format_message(message.role, _message_text(message)) for message in prompt]

    @abstractmethod
    async def _initialize_state(
        self, *, prompt: list[mcp_types.PromptMessage]
    ) -> RunState[MessageT]:
        """Build fresh run state from the prompt turns (use ``self._initial_messages``)."""

    @abstractmethod
    async def get_response(
        self,
        state: RunState[MessageT],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentStep:
        """Call the provider API and return the model's turn as an ``AgentStep``.

        The loop stamps ``started_at``/``model`` fallbacks and records it;
        a failed call is an ``AgentStep`` with ``error`` set and ``done=True``.
        """

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


__all__ = ["RunState", "ToolAgent"]
