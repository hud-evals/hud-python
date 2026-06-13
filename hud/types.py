"""Universal SDK shapes, including the trajectory contract.

The trajectory contract: a ``Trace`` is an ordered collection of ``Step``s,
and recording a step (:meth:`Trace.record`) ships it to the platform as one
schema-tagged span. ``Step`` here is the shared skeleton every agent family
and the run harness speak — ordering, source, timing, error — and the
harness records its own steps directly (``user`` prompt turns, ``task``
lifecycle calls, ``system`` errors).

Agent families layer their payloads on top by subclassing :class:`Step` —
the tool-agent family adds LLM responses and tool calls in
:mod:`hud.agents.types` under the ``hud.step.v1`` schema; other families
(e.g. robot) bring their own payload fields under their own ``schema_tag``
and inherit the transport. The platform's serializer registry dispatches on
the schema tag, so each family decodes losslessly without this module or the
telemetry pipe (:mod:`hud.telemetry`) knowing any payload shape.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias, TypeVar, cast

import mcp.types as types
from mcp.types import CallToolRequestParams, CallToolResult
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from hud.telemetry.context import get_current_trace_id
from hud.telemetry.exporter import queue_span
from hud.telemetry.span import (
    PAYLOAD_ATTRIBUTE,
    SCHEMA_ATTRIBUTE,
    TASK_RUN_ID_ATTRIBUTE,
    Span,
    new_span_id,
    normalize_trace_id,
)
from hud.utils.serialization import JsonObject, JsonValue
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from collections.abc import Callable

    from hud.agents.claude import ClaudeAgent
    from hud.agents.gemini import GeminiAgent
    from hud.agents.openai import OpenAIAgent
    from hud.agents.openai_compatible import OpenAIChatAgent
    from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig

    AgentClass: TypeAlias = type[ClaudeAgent | GeminiAgent | OpenAIAgent | OpenAIChatAgent]
    AgentConfigClass: TypeAlias = type[
        ClaudeConfig | GeminiConfig | OpenAIConfig | OpenAIChatConfig
    ]

T = TypeVar("T")


class AgentType(str, Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"

    @property
    def cls(self) -> AgentClass:
        match self:
            case AgentType.CLAUDE:
                from hud.agents import ClaudeAgent

                return ClaudeAgent
            case AgentType.OPENAI:
                from hud.agents import OpenAIAgent

                return OpenAIAgent
            case AgentType.GEMINI:
                from hud.agents import GeminiAgent

                return GeminiAgent
            case AgentType.OPENAI_COMPATIBLE:
                from hud.agents import OpenAIChatAgent

                return OpenAIChatAgent

    @property
    def config_cls(self) -> AgentConfigClass:
        """Get config class without importing agent (avoids SDK dependency)."""
        from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig

        match self:
            case AgentType.CLAUDE:
                return ClaudeConfig
            case AgentType.OPENAI:
                return OpenAIConfig
            case AgentType.GEMINI:
                return GeminiConfig
            case AgentType.OPENAI_COMPATIBLE:
                return OpenAIChatConfig

    @property
    def gateway_provider(self) -> str:
        """Default provider client used when this agent type is a gateway shortcut."""
        match self:
            case AgentType.CLAUDE:
                return "anthropic"
            case AgentType.OPENAI:
                return "openai"
            case AgentType.GEMINI:
                return "gemini"
            case AgentType.OPENAI_COMPATIBLE:
                return "openai"

    @classmethod
    def of(cls, agent: object) -> AgentType | None:
        """The gateway agent type *agent* is an instance of, or ``None``.

        Reverse of :attr:`cls`. Provider extras (anthropic, google-genai, ...)
        may be uninstalled, so importing a type's agent class can fail; that
        simply means *agent* is not that type. ``None`` for a custom ``Agent``
        subclass that is not one of the gateway shortcuts.
        """
        for agent_type in cls:
            with contextlib.suppress(Exception):
                if isinstance(agent, agent_type.cls):
                    return agent_type
        return None


class MCPToolCall(CallToolRequestParams):
    """A tool call."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier for reference
    annotation: str | None = None  # Optional explanation of why this action is taken
    provider_name: str | None = None  # Original provider tool name when it differs from MCP name

    def __str__(self) -> str:
        """Format tool call as plain text."""
        args_str = ""
        if self.arguments:
            try:
                args_str = json.dumps(self.arguments, separators=(",", ":"))
                if len(args_str) > 60:
                    args_str = args_str[:57] + "..."
            except (TypeError, ValueError):
                args_str = str(self.arguments)[:60]

        s = f"→ {self.name}({args_str})"
        if self.annotation:
            s += f"  # {self.annotation}"
        return s

    def __rich__(self) -> str:
        """Rich representation with color formatting."""
        from rich.markup import escape

        from hud.utils.hud_console import hud_console

        s = hud_console.format_tool_call(self.name, self.arguments)
        if self.annotation:
            s += f"  [bright_black]# {escape(self.annotation)}[/bright_black]"
        return s


class MCPToolResult(CallToolResult):
    """A tool result with optional call_id for correlation."""

    call_id: str | None = None  # For correlating with provider-specific tool call IDs

    def _get_content_summary(self) -> str:
        """Extract a summary of the content."""
        # Extract content summary
        content_summary = ""
        if self.content:
            for block in self.content:
                if isinstance(block, types.TextContent):
                    # Get first line or truncate
                    text = block.text.strip()
                    first_line = text.split("\n")[0] if "\n" in text else text
                    content_summary = first_line
                    break
                elif isinstance(block, types.ImageContent):
                    content_summary = "📷 Image"
                    break

        # Or use structured content if no text content
        if not content_summary and self.structuredContent:
            try:
                content_summary = json.dumps(self.structuredContent, separators=(",", ":"))
            except (TypeError, ValueError):
                content_summary = str(self.structuredContent)

        return content_summary

    def __str__(self) -> str:
        """Format tool result as plain text for compatibility."""
        content_summary = self._get_content_summary()

        # Plain text format with unicode symbols
        if self.isError:
            return f"✗ {content_summary}"
        else:
            return f"✓ {content_summary}"

    def __rich__(self) -> str:
        """Rich representation with color formatting."""
        from hud.utils.hud_console import hud_console

        content_summary = self._get_content_summary()
        return hud_console.format_tool_result(content_summary, self.isError)


# -----------------------------------------------------------------------------
# Trajectory contract
# -----------------------------------------------------------------------------

#: Schema tag of the core step stream (the tool-agent family shares it).
STEP_SCHEMA = "hud.step.v1"

StepSource: TypeAlias = Literal["user", "agent", "tool", "task", "subagent", "system"]


class TaskCall(BaseModel):
    """The task-lifecycle RPC a ``task`` step records.

    ``setup`` is ``tasks.start`` (result carries the opening prompt payload);
    ``evaluate`` is ``tasks.grade`` (result carries the evaluation dict).
    """

    phase: Literal["setup", "evaluate"]
    name: str
    arguments: JsonValue = None
    result: JsonValue = None


class Step(BaseModel):
    """One ordered interaction unit in a task run — the shared skeleton.

    Carries what every family and the harness need: position, ``source``,
    timing, ``error``, and the harness payloads — ``messages`` (user/system
    prompt turns, kept as structured ``PromptMessage``s so multimodal content
    is preserved) and ``task_call`` (task lifecycle RPC). Agent families
    subclass this with their own payload fields (e.g.
    :class:`hud.agents.types.AgentStep`) and, for a new wire schema, their
    own ``schema_tag``.
    """

    #: Schema tag this step ships under; families with their own wire schema
    #: override it (the platform's serializer registry dispatches on it).
    schema_tag: ClassVar[str] = STEP_SCHEMA

    # Sequential position in the trace, assigned by ``Trace`` (1-based).
    step_id: int | None = None
    source: StepSource

    messages: list[types.PromptMessage] = Field(default_factory=list[types.PromptMessage])
    task_call: TaskCall | None = None

    error: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    extra: JsonObject = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    def emit(self) -> None:
        """Queue this step for export as a span tagged with its schema.

        The payload is the step's own dump, so family subclasses ship their
        full payload under their ``schema_tag`` with no extra wiring. No-op
        without an ambient trace context (nothing to attribute it to).

        :meth:`Trace.record` calls this for every recorded step; calling it
        directly is for steps that report outside their own local trace
        (e.g. a ``SubagentStep`` reporting a sub-rollout to the enclosing
        trace context).
        """
        task_run_id = get_current_trace_id()
        if not task_run_id:
            return

        now = now_iso()
        payload = cast("JsonObject", self.model_dump(mode="json", exclude_none=True))
        span = Span(
            name=f"step.{self.source}",
            trace_id=normalize_trace_id(task_run_id),
            span_id=new_span_id(),
            start_time=self.started_at or now,
            end_time=self.ended_at or now,
            status_code="ERROR" if self.error else "OK",
            status_message=self.error,
            attributes={
                SCHEMA_ATTRIBUTE: self.schema_tag,
                TASK_RUN_ID_ATTRIBUTE: task_run_id,
                PAYLOAD_ATTRIBUTE: payload,
            },
        )
        queue_span(span.model_dump(mode="json"))


TraceStatus: TypeAlias = Literal["completed", "error", "cancelled"]


class Trace(BaseModel):
    """The agent's trajectory for one rollout — ordered ``Step``s that ship as spans.

    A serializable list of ordered ``Step``s plus the run summary: ``status``
    and the final ``content`` (the graded answer). Everything else the summary
    exposes is *derived* from the steps — the steps are the record, the
    summary is a view. :meth:`final` and :meth:`collect` are the two derivation
    shapes (newest answer wins / every answer in order); ``error`` is a view
    built on them, and family-specific reads use the same queries with family
    vocabulary at the call site (e.g. the tool-agent family's reply citations,
    or its token samples for an external trainer). The unit of training
    data — family payloads on the steps carry the trainable record.
    :meth:`record` is the single write path: it appends *and* streams the
    step to the platform. The task lifecycle (prompt, reward, evaluation)
    and the live connection live on ``Run``, not here.

    ``steps`` hold family subclasses at runtime; dumps serialize each step by
    its runtime type so family payloads survive serialization.
    """

    steps: list[SerializeAsAny[Step]] = Field(default_factory=list[Step])

    status: TraceStatus | None = None
    content: str | None = Field(default=None)

    # Trajectory metadata that has no structured home (provider session info,
    # external-SDK run stats). Never load-bearing for the platform.
    extra: JsonObject = Field(default_factory=dict)

    # Keys the server-side-collected trajectory; None for eval-only runs.
    trace_id: str | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")

    def final(self, get: Callable[[Step], T | None]) -> T | None:
        """The newest step's answer to *get* — the finalized-field query.

        Asks steps newest-first and returns the first non-``None`` answer
        (``None`` from a step means "no answer here", so falsy answers like
        ``""`` or ``[]`` still win). ``None`` when no step answers. Derived
        summary fields are views on this — see ``error``.
        """
        return next(
            (value for step in reversed(self.steps) if (value := get(step)) is not None),
            None,
        )

    def collect(self, get: Callable[[Step], T | None]) -> list[T]:
        """Every step's answer to *get*, in step order — the gathering query.

        Steps answering ``None`` are skipped. Family-specific reads keep
        their vocabulary at the call site, e.g. the tool-agent family's
        training samples::

            trace.collect(lambda s: s.sample if isinstance(s, AgentStep) else None)
        """
        return [value for step in self.steps if (value := get(step)) is not None]

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    @property
    def error(self) -> str | None:
        """The most recent step error, if any (errors live on steps)."""
        return self.final(lambda step: step.error)

    @model_validator(mode="after")
    def _number_steps(self) -> Trace:
        for index, step in enumerate(self.steps, start=1):
            step.step_id = index
        return self

    def record(self, step: Step) -> None:
        """Append one step and stream it to the platform — the single write path.

        Numbers the step, stamps ``ended_at`` when unset (a step ends when
        it's recorded), appends it, and emits it as a span (a no-op without
        an ambient trace context). Callers stamp ``started_at`` themselves
        when the step wraps awaited work — only the call site knows when
        that began.
        """
        step.step_id = len(self.steps) + 1
        if step.ended_at is None:
            step.ended_at = now_iso()
        self.steps.append(step)
        step.emit()

    def __len__(self) -> int:
        return len(self.steps)


__all__ = [
    "STEP_SCHEMA",
    "AgentType",
    "JsonObject",
    "JsonValue",
    "MCPToolCall",
    "MCPToolResult",
    "Step",
    "StepSource",
    "TaskCall",
    "Trace",
    "TraceStatus",
]
