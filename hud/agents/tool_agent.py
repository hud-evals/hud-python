"""Catalog-driven provider tool-call agents."""

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
from hud.agents.tools.base import AgentTool, provider_tool_name
from hud.agents.tools.mcp import MCPTool
from hud.agents.tools.rfb import RFBTool
from hud.agents.tools.ssh import SSHInfrastructureErrorResult, SSHTool
from hud.agents.types import AgentStep, ToolStep
from hud.capabilities import MCPClient, RFBClient
from hud.capabilities.ssh import SSHConnectionError
from hud.types import MCPToolCall, MCPToolResult, Step, StopCondition
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from hud.agents.types import ToolAgentConfig
    from hud.capabilities import CapabilityClient
    from hud.eval.run import Run
    from hud.utils.serialization import JsonObject

logger = logging.getLogger(__name__)

# Output-token-cap finish reasons: OpenAI chat "length", Responses "max_output_tokens",
# Claude "max_tokens", Gemini "MAX_TOKENS".
TRUNCATION_FINISH_REASONS = frozenset({"length", "max_output_tokens", "max_tokens", "MAX_TOKENS"})
MAX_CONSECUTIVE_SSH_FAILURES = 3

MessageT = TypeVar("MessageT")
ConfigT = TypeVar("ConfigT", bound="ToolAgentConfig")


class DegenerateTurnError(Exception):
    """A model sample was unusable and was not committed to the run state.

    Raised by ``get_response`` implementations that reject a degenerate
    response before recording it. The loop discards the turn and samples a
    fresh one; each discarded turn still consumes a step from ``max_steps``,
    and a degenerate final step fails the run.
    """


def _message_text(message: mcp_types.PromptMessage) -> str:
    content = message.content
    return content.text if isinstance(content, mcp_types.TextContent) else ""


def _block_text(block: mcp_types.ContentBlock) -> str | None:
    match block:
        case mcp_types.TextContent():
            return block.text
        case mcp_types.EmbeddedResource(resource=mcp_types.TextResourceContents() as resource):
            return resource.text
        case _:
            return None


def _with_text(block: mcp_types.ContentBlock, text: str) -> mcp_types.ContentBlock:
    match block:
        case mcp_types.TextContent():
            return block.model_copy(update={"text": text})
        case mcp_types.EmbeddedResource(resource=mcp_types.TextResourceContents() as resource):
            return block.model_copy(update={"resource": resource.model_copy(update={"text": text})})
        case _:
            raise TypeError(f"{type(block).__name__} carries no text")


def _truncation_notice(omitted: int, total: int, limit: int) -> str:
    return (
        f"\n\n[... {omitted:,} of {total:,} characters omitted: this tool result exceeded the "
        f"{limit:,}-character limit. Read large files in smaller line ranges, or narrow "
        f"command output with grep, head, tail, or sed -n ...]\n\n"
    )


def _bound_tool_result(result: MCPToolResult, limit: int) -> MCPToolResult:
    """Fit a tool result's model-visible text within ``limit`` characters.

    Text blocks and embedded text resources count toward the limit, as does
    ``structuredContent`` (providers serialize it to JSON for the model). An
    oversized result drops ``structuredContent``, keeping its JSON as text only
    when the result has no text of its own, then keeps the head and tail of the
    text around a truncation notice. Images and binary resources pass through.
    """
    texts = [_block_text(block) for block in result.content]
    structured = (
        None
        if result.structuredContent is None
        else json.dumps(result.structuredContent, default=str)
    )
    total = sum(len(text) for text in texts if text is not None) + len(structured or "")
    if total <= limit:
        return result

    content: list[mcp_types.ContentBlock] = list(result.content)
    if structured is not None and all(text is None for text in texts):
        content.append(mcp_types.TextContent(type="text", text=structured))
        texts.append(structured)
    total = sum(len(text) for text in texts if text is not None)
    if total <= limit:
        return result.model_copy(update={"content": content, "structuredContent": None})

    keep = limit - len(_truncation_notice(total, total, limit))
    head_left, tail_left = (keep + 1) // 2, keep // 2
    heads: list[int] = []
    for text in texts:
        taken = min(len(text or ""), head_left)
        heads.append(taken)
        head_left -= taken
    tails = [0] * len(texts)
    for index in reversed(range(len(texts))):
        taken = min(len(texts[index] or "") - heads[index], tail_left)
        tails[index] = taken
        tail_left -= taken
    cut = next(
        index for index, text in enumerate(texts) if text is not None and heads[index] < len(text)
    )
    notice = _truncation_notice(total - keep, total, limit)

    bounded: list[mcp_types.ContentBlock] = []
    for index, (block, text) in enumerate(zip(content, texts, strict=True)):
        if text is None:
            bounded.append(block)
            continue
        kept = (
            text[: heads[index]]
            + (notice if index == cut else "")
            + text[len(text) - tails[index] :]
        )
        if kept:
            bounded.append(_with_text(block, kept))
    return result.model_copy(update={"content": bounded, "structuredContent": None})


@dataclass
class RunState(Generic[MessageT]):
    """Provider messages and tools for one run."""

    messages: list[MessageT] = field(default_factory=list[MessageT])
    tools: dict[str, AgentTool[Any]] = field(default_factory=dict[str, AgentTool[Any]])
    params: list[Any] = field(default_factory=list[Any])


class ToolAgent(Agent[ConfigT], Generic[MessageT, ConfigT]):
    """Catalog-driven provider tool-call loop."""

    tool_catalog: ClassVar[tuple[type[AgentTool[Any]], ...]] = ()
    #: Capability-client types this agent can drive (derived from the catalog).
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
        manifest = run.client.manifest
        if manifest is not None:
            wanted = {client.protocol for client in type(self).clients}
            for cap in manifest.bindings:
                if cap.protocol not in wanted:
                    continue
                if cap.protocol != MCPClient.protocol and cap.protocol in opened_protocols:
                    continue
                connections[cap.name] = await run.client.open(cap.name)
                opened_protocols.add(cap.protocol)
        state = await self._initialize_state(prompt=run.prompt_messages)
        state.tools, state.params = await self._build_tools(connections)
        await self._loop(run, state)

    async def _build_tools(
        self,
        connections: dict[str, CapabilityClient],
    ) -> tuple[dict[str, AgentTool[Any]], list[Any]]:
        """Build the (tools, params) for one run from the given open connections."""
        tools: dict[str, AgentTool[Any]] = {}
        params: list[Any] = []
        model = self.config.model

        mcp_clients = [c for c in connections.values() if isinstance(c, MCPClient)]
        mcp_lists = await asyncio.gather(*(c.list_tools() for c in mcp_clients))
        mcp_by_client: dict[MCPClient, list[mcp_types.Tool]] = dict(
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
                    if issubclass(tool_cls, RFBTool):
                        assert isinstance(client, RFBClient)
                        tool = tool_cls(
                            spec=spec,
                            client=client,
                            screenshot_encoding=self.config.screenshot_encoding,
                        )
                    else:
                        tool = tool_cls(spec=spec, client=client)
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
                run.record(step)
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
                            if isinstance(follow_up.content, mcp_types.TextContent)
                            else ""
                        )
                        state.messages.append(self._format_user_text(text))
                        run.record(Step(source="user", messages=[follow_up]))
                        continue
                    break

                if (stopped := self._stop_condition(step)) is not None:
                    break

                for call in step.tool_calls:
                    call_started_at = now_iso()
                    result = _bound_tool_result(
                        await self._dispatch_call(call, state),
                        self.config.max_tool_result_chars,
                    )
                    run.record(ToolStep(call=call, result=result, started_at=call_started_at))
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
                        run.record(Step(source="system", error=error))
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
            run.record(Step(source="system", error=str(exc)))

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
        call: MCPToolCall,
        state: RunState[MessageT],
    ) -> MCPToolResult:
        if isinstance(call.arguments, str):
            return MCPToolResult(
                content=[
                    mcp_types.TextContent(
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
            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text=f"unknown tool: {call.name!r}")],
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
                    return MCPToolResult(
                        content=[
                            mcp_types.TextContent(
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
                content=[mcp_types.TextContent(type="text", text=f"tool error: {exc}")],
                isError=True,
            )
        except Exception as exc:
            logger.exception("tool %s failed", call.name)
            return MCPToolResult(
                content=[mcp_types.TextContent(type="text", text=f"tool error: {exc}")],
                isError=True,
            )

    def _initial_messages(self, prompt: list[mcp_types.PromptMessage]) -> list[MessageT]:
        """Map normalized prompt turns onto provider messages."""
        return [self._format_message(message.role, _message_text(message)) for message in prompt]

    def _format_user_text(self, text: str) -> MessageT:
        """Wrap a plain text string as a provider user message."""
        return self._format_message("user", text)

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

        The loop stamps ``started_at``/``model`` fallbacks, records the step,
        and raises its error if present.
        """

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
