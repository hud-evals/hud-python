"""Catalog-driven provider tool-call agents."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from hud.agents.base import Agent
from hud.agents.misc.response_automation import auto_respond
from hud.agents.tools.base import AgentTool, provider_tool_name
from hud.agents.tools.mcp import MCPTool
from hud.agents.tools.ssh import SSHInfrastructureErrorResult, SSHTool
from hud.agents.types import AgentStep, ToolStep
from hud.capabilities import MCPClient, open_client
from hud.capabilities.ssh import SSHConnectionError
from hud.types import PromptMessage, Step, StopCondition, TextContent, Tool, ToolCall, ToolResult
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.agents.types import ToolAgentConfig
    from hud.capabilities import CapabilityClient
    from hud.eval.run import Run
    from hud.utils.serialization import JsonObject

logger = logging.getLogger(__name__)

# Output-token-cap finish reasons: OpenAI chat "length", Responses "max_output_tokens",
# Claude "max_tokens", Gemini "MAX_TOKENS", and Gemini "CONTINUATION" (a per-request
# cap hit by a turn that was not continued).
TRUNCATION_FINISH_REASONS = frozenset(
    {"length", "max_output_tokens", "max_tokens", "MAX_TOKENS", "CONTINUATION"}
)
MAX_CONSECUTIVE_SSH_FAILURES = 3

MessageT = TypeVar("MessageT")
ConfigT = TypeVar("ConfigT", bound="ToolAgentConfig")


class DegenerateTurnError(Exception):
    """Raised by ``get_response`` for an unusable sample it did not commit to the run state.

    The loop resamples; each discarded turn still spends a step, and one on the last step fails
    the run.
    """


def _message_text(message: PromptMessage) -> str:
    content = message.content
    return content.text if isinstance(content, TextContent) else ""


@dataclass
class RunState(Generic[MessageT]):
    """Provider messages and tools for one run."""

    messages: list[MessageT] = field(default_factory=list[MessageT])
    tools: dict[str, AgentTool[Any]] = field(default_factory=dict[str, AgentTool[Any]])
    params: list[Any] = field(default_factory=list[Any])


class ToolAgent(Agent[ConfigT], Generic[MessageT, ConfigT]):
    """Catalog-driven provider tool-call loop."""

    tool_catalog: ClassVar[tuple[type[AgentTool[Any]], ...]] = ()
    # Capability-client types this agent can drive (derived from the catalog).
    clients: ClassVar[tuple[type[CapabilityClient], ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "tool_catalog" in cls.__dict__:
            seen: dict[type[CapabilityClient], None] = {}
            for t in cls.tool_catalog:
                seen.setdefault(t.client_type, None)
            cls.clients = tuple(seen.keys())

    def dump(self) -> JsonObject:
        if self.config.model_client is not None:
            raise ValueError(
                "a custom model_client cannot be serialized; use create_agent(model, ...) "
                "so the agent rebuilds the gateway client, or run the agent loop locally "
                "with HUDRuntime() / LocalRuntime"
            )
        return super().dump()

    async def __call__(self, run: Run) -> None:
        connections: dict[str, CapabilityClient] = {}
        opened_protocols: set[str] = set()
        wanted = {client.protocol for client in type(self).clients}
        async with contextlib.AsyncExitStack() as opened:
            for cap in run.capabilities:
                if cap.protocol not in wanted:
                    continue
                if cap.protocol != MCPClient.protocol and cap.protocol in opened_protocols:
                    continue
                connections[cap.name] = await opened.enter_async_context(open_client(cap))
                opened_protocols.add(cap.protocol)
            state = await self._initialize_state(prompt=run.prompt)
            state.tools, state.params = await self._build_tools(connections)
            await self._loop(run, state)

    async def _build_tools(
        self,
        connections: dict[str, CapabilityClient],
    ) -> tuple[dict[str, AgentTool[Any]], list[Any]]:
        tools: dict[str, AgentTool[Any]] = {}
        params: list[Any] = []
        model = self.config.model

        mcp_clients = [c for c in connections.values() if isinstance(c, MCPClient)]
        mcp_lists = await asyncio.gather(*(c.list_tools() for c in mcp_clients))
        mcp_by_client: dict[MCPClient, list[Tool]] = dict(
            zip(mcp_clients, mcp_lists, strict=True),
        )
        qualify_mcp_names = len(mcp_clients) > 1

        for tool_cls in type(self).tool_catalog:
            spec = tool_cls.default_spec(model)
            if spec is None:
                continue
            for connection_name, client in connections.items():
                if not isinstance(client, tool_cls.client_type):
                    continue
                if issubclass(tool_cls, MCPTool):
                    assert isinstance(client, MCPClient)
                    for mt in mcp_by_client[client]:
                        qualified_name = (
                            f"{connection_name}__{mt.name}" if qualify_mcp_names else mt.name
                        )
                        tool = tool_cls(
                            spec=spec,
                            client=client,
                            mcp_tool=mt,
                            provider_name=provider_tool_name(qualified_name),
                        )
                        if tool.provider_name in tools:
                            raise ValueError(
                                "MCP tool name collision after qualification: "
                                f"{tool.provider_name!r}"
                            )
                        tools[tool.provider_name] = tool
                        params.append(tool.to_params())
                else:
                    tool = tool_cls.from_config(spec=spec, client=client, config=self.config)
                    tools[tool.provider_name] = tool
                    params.append(tool.to_params())

        params.extend(
            hosted.to_params()
            for hosted in self.config.hosted_tools
            if hosted.supports_model(model)
        )

        return tools, params

    async def _loop(
        self,
        run: Run,
        state: RunState[MessageT],
    ) -> None:
        trace = run.trace
        try:
            step: AgentStep | None = None
            hit_max = False
            stopped: StopCondition | None = None
            consecutive_ssh_failures = 0

            for turn in range(1, self.config.max_steps + 1):
                logger.info("step %d/%d", turn, self.config.max_steps)
                started_at = now_iso()
                try:
                    step = await self.get_response(
                        state,
                        system_prompt=self.config.system_prompt,
                        citations_enabled=self.config.citations_enabled,
                    )
                except DegenerateTurnError as exc:
                    if turn == self.config.max_steps:
                        raise
                    logger.warning("Discarded degenerate turn: %s", exc)
                    continue
                step.started_at = step.started_at or started_at
                step.model = step.model or self.config.model
                run.trace.record(step)
                if step.error:
                    stopped = step.stop_reason
                    break

                if step.tool_calls:
                    logger.info("  → %s", ", ".join(c.name for c in step.tool_calls))

                if step.done or not step.tool_calls:
                    follow_up = await auto_respond(step.content, enabled=self.config.auto_respond)
                    if follow_up is not None:
                        text = (
                            follow_up.content.text
                            if isinstance(follow_up.content, TextContent)
                            else ""
                        )
                        state.messages.append(self._format_user_text(text))
                        run.trace.record(Step(source="user", messages=[follow_up]))
                        continue
                    break

                if (stopped := self._stop_condition(step)) is not None:
                    break

                for call in step.tool_calls:
                    call_started_at = now_iso()
                    result = await self._dispatch_call(call, state)
                    run.trace.record(ToolStep(call=call, result=result, started_at=call_started_at))
                    msg = self._format_result(call, result, state)
                    if isinstance(msg, list):
                        state.messages.extend(msg)
                    elif msg is not None:
                        state.messages.append(cast("MessageT", msg))

                    if isinstance(result, SSHInfrastructureErrorResult):
                        consecutive_ssh_failures += 1
                    else:
                        consecutive_ssh_failures = 0
                    if consecutive_ssh_failures >= MAX_CONSECUTIVE_SSH_FAILURES:
                        error = (
                            "SSH tool failure limit reached "
                            f"after {MAX_CONSECUTIVE_SSH_FAILURES} consecutive errors"
                        )
                        trace.content = step.content
                        trace.status = "error"
                        run.trace.record(Step(source="system", error=error))
                        return

                if turn == self.config.max_steps:
                    hit_max = True

            trace.content = step.content if step else None
            trace.status = "error" if step is not None and step.error else "completed"
            if stopped is not None:
                trace.stop_reason = stopped
            elif step is not None and step.error:
                trace.stop_reason = None
            elif hit_max:
                trace.stop_reason = "max_steps"
            elif step is not None and step.finish_reason in TRUNCATION_FINISH_REASONS:
                trace.stop_reason = "length"
            else:
                trace.stop_reason = "done"
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:
            logger.exception("ToolAgent loop failed")
            trace.status = "error"
            run.trace.record(Step(source="system", error=str(exc)))

    def _stop_condition(self, step: AgentStep) -> StopCondition | None:
        """The first configured stop condition this turn trips, if any."""
        stop_on = self.config.stop_on
        if "length" in stop_on and step.finish_reason in TRUNCATION_FINISH_REASONS:
            return "length"
        if "malformed_tool_call" in stop_on and any(
            isinstance(call.arguments, str) for call in step.tool_calls
        ):
            return "malformed_tool_call"
        return None

    async def _dispatch_call(
        self,
        call: ToolCall,
        state: RunState[MessageT],
    ) -> ToolResult:
        if isinstance(call.arguments, str):
            return ToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"the arguments for this {call.name!r} call arrived incomplete "
                            f"(cut off mid-generation or invalid JSON), so it was not executed. "
                            f"Received: {call.arguments[:200]!r}. Re-issue the call in full."
                        ),
                    )
                ],
                isError=True,
            )
        tool = state.tools.get(call.name)
        if tool is None:
            return ToolResult(
                content=[TextContent(type="text", text=f"unknown tool: {call.name!r}")],
                isError=True,
            )
        args = call.arguments or {}
        try:
            if isinstance(tool, SSHTool) and self.config.tool_timeout_seconds is not None:
                timeout = self.config.tool_timeout_seconds
                deadline = asyncio.timeout(timeout)
                try:
                    async with deadline:
                        return await tool.execute(args)
                except TimeoutError:
                    if not deadline.expired():
                        raise
                    return ToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=(
                                    f"{call.name} timed out after {timeout:g}s; "
                                    "retry with a shorter command"
                                ),
                            )
                        ],
                        isError=True,
                    )
            return await tool.execute(args)
        except (TimeoutError, asyncio.CancelledError):
            raise
        except SSHConnectionError as exc:
            logger.exception("tool %s lost its SSH connection", call.name)
            return SSHInfrastructureErrorResult(
                content=[TextContent(type="text", text=f"tool error: {exc}")],
                isError=True,
            )
        except Exception as exc:
            logger.exception("tool %s failed", call.name)
            return ToolResult(
                content=[TextContent(type="text", text=f"tool error: {exc}")],
                isError=True,
            )

    def _initial_messages(self, prompt: list[PromptMessage]) -> list[MessageT]:
        return [self._format_message(message.role, _message_text(message)) for message in prompt]

    def _format_user_text(self, text: str) -> MessageT:
        return self._format_message("user", text)

    @abstractmethod
    async def _initialize_state(self, *, prompt: list[PromptMessage]) -> RunState[MessageT]:
        """Build fresh run state from the prompt turns (use ``self._initial_messages``)."""

    @abstractmethod
    async def get_response(
        self,
        state: RunState[MessageT],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentStep:
        """Call the provider and return the model's turn.

        The loop fills an unset ``started_at`` and ``model``, records the step, and ends the run
        on its ``error``; raise ``DegenerateTurnError`` to discard the sample instead.
        """

    @abstractmethod
    def _format_message(self, role: str, text: str) -> MessageT:
        """Wrap text as a provider message of the given role (``user``/``assistant``)."""

    @abstractmethod
    def _format_result(
        self,
        call: ToolCall,
        result: ToolResult,
        state: RunState[MessageT],
    ) -> MessageT | list[MessageT] | None:
        """Convert a tool result into one or more provider messages, or None to skip."""
