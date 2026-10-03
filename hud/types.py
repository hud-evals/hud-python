"""Universal SDK shapes, including the trajectory contract."""

from __future__ import annotations

import contextlib
import json
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, TypeVar, cast

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializeAsAny,
    model_validator,
)

from hud.graders.results import SubScore
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
    from collections.abc import Callable, Sequence

    from hud.agents.base import Agent
    from hud.agents.types import AgentConfig

T = TypeVar("T")


class AgentType(StrEnum):
    CLAUDE = "claude"
    CLAUDE_CLI = "claude_cli"
    CODEX_CLI = "codex_cli"
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"

    @property
    def is_cli(self) -> bool:
        return self in (AgentType.CLAUDE_CLI, AgentType.CODEX_CLI)

    @property
    def cls(self) -> type[Agent]:
        match self:
            case AgentType.CLAUDE:
                from hud.agents import ClaudeAgent

                return ClaudeAgent
            case AgentType.CLAUDE_CLI:
                from hud.agents import ClaudeCLIAgent

                return ClaudeCLIAgent
            case AgentType.CODEX_CLI:
                from hud.agents import CodexCLIAgent

                return CodexCLIAgent
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
    def config_cls(self) -> type[AgentConfig]:
        """Get config class without importing agent (avoids SDK dependency)."""
        from hud.agents.types import (
            ClaudeCLIConfig,
            ClaudeConfig,
            CodexCLIConfig,
            GeminiConfig,
            OpenAIChatConfig,
            OpenAIConfig,
        )

        match self:
            case AgentType.CLAUDE:
                return ClaudeConfig
            case AgentType.CLAUDE_CLI:
                return ClaudeCLIConfig
            case AgentType.CODEX_CLI:
                return CodexCLIConfig
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
            case AgentType.CLAUDE_CLI:
                return "anthropic"
            case AgentType.CODEX_CLI:
                return "openai"
            case AgentType.OPENAI:
                return "openai"
            case AgentType.GEMINI:
                return "gemini"
            case AgentType.OPENAI_COMPATIBLE:
                return "openai"

    @classmethod
    def of(cls, agent: object) -> AgentType | None:
        """The registered agent type ``agent`` is an instance of, or ``None``.

        A type whose agent extra isn't installed is skipped.
        """
        for agent_type in cls:
            with contextlib.suppress(ImportError):
                if isinstance(agent, agent_type.cls):
                    return agent_type
        return None


# The shapes traces record keep MCP's camelCase field names (mimeType, isError), the
# wire format the platform stores; MCP clients and servers convert at their boundary.

Role: TypeAlias = Literal["user", "assistant"]


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Annotations(_WireModel):
    audience: list[Role] | None = None
    priority: float | None = Field(default=None, ge=0.0, le=1.0)


class TextContent(_WireModel):
    type: Literal["text"]
    text: str
    annotations: Annotations | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ImageContent(_WireModel):
    type: Literal["image"]
    data: str
    mimeType: str
    annotations: Annotations | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class AudioContent(_WireModel):
    type: Literal["audio"]
    data: str
    mimeType: str
    annotations: Annotations | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class Icon(_WireModel):
    src: str
    mimeType: str | None = None
    sizes: list[str] | None = None


class ResourceLink(_WireModel):
    type: Literal["resource_link"]
    name: str
    title: str | None = None
    uri: AnyUrl
    description: str | None = None
    mimeType: str | None = None
    size: int | None = None
    icons: list[Icon] | None = None
    annotations: Annotations | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class TextResourceContents(_WireModel):
    uri: AnyUrl
    mimeType: str | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")
    text: str


class BlobResourceContents(_WireModel):
    uri: AnyUrl
    mimeType: str | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")
    blob: str


class EmbeddedResource(_WireModel):
    type: Literal["resource"]
    resource: TextResourceContents | BlobResourceContents
    annotations: Annotations | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


ContentBlock: TypeAlias = (
    TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource
)


class PromptMessage(_WireModel):
    role: Role
    content: ContentBlock


def text_of(messages: Sequence[PromptMessage]) -> str:
    """The text turns of ``messages`` joined by blank lines; images and resources are dropped."""
    return "\n\n".join(
        message.content.text
        for message in messages
        if isinstance(message.content, TextContent) and message.content.text
    )


class ToolAnnotations(_WireModel):
    title: str | None = None
    readOnlyHint: bool | None = None
    destructiveHint: bool | None = None
    idempotentHint: bool | None = None
    openWorldHint: bool | None = None


class Tool(_WireModel):
    """A tool an environment offers the agent."""

    name: str
    title: str | None = None
    description: str | None = None
    inputSchema: dict[str, Any]
    outputSchema: dict[str, Any] | None = None
    icons: list[Icon] | None = None
    annotations: ToolAnnotations | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")
    execution: dict[str, Any] | None = None


class ToolCall(_WireModel):
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")
    name: str
    # A str is the raw provider string that didn't parse as JSON; dispatch answers it
    # with an error result and never executes it.
    arguments: dict[str, Any] | str | None = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    annotation: str | None = None  # Optional explanation of why this action is taken
    provider_name: str | None = None  # Original provider tool name when it differs from MCP name

    def __str__(self) -> str:
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
        from rich.markup import escape

        from hud.utils.hud_console import hud_console

        s = hud_console.format_tool_call(self.name, self.arguments)
        if self.annotation:
            s += f"  [bright_black]# {escape(self.annotation)}[/bright_black]"
        return s


class ToolResult(_WireModel):
    """A tool result with optional call_id for correlation."""

    meta: dict[str, Any] | None = Field(default=None, alias="_meta")
    content: list[ContentBlock]
    structuredContent: dict[str, Any] | None = None
    isError: bool = False
    call_id: str | None = None  # For correlating with provider-specific tool call IDs

    def _get_content_summary(self) -> str:
        content_summary = ""
        if self.content:
            for block in self.content:
                if isinstance(block, TextContent):
                    text = block.text.strip()
                    first_line = text.split("\n")[0] if "\n" in text else text
                    content_summary = first_line
                    break
                elif isinstance(block, ImageContent):
                    content_summary = "📷 Image"
                    break

        if not content_summary and self.structuredContent:
            try:
                content_summary = json.dumps(self.structuredContent, separators=(",", ":"))
            except (TypeError, ValueError):
                content_summary = str(self.structuredContent)

        return content_summary

    def __str__(self) -> str:
        content_summary = self._get_content_summary()

        if self.isError:
            return f"✗ {content_summary}"
        else:
            return f"✓ {content_summary}"

    def __rich__(self) -> str:
        from hud.utils.hud_console import hud_console

        content_summary = self._get_content_summary()
        return hud_console.format_tool_result(content_summary, self.isError)


# Schema tag of the core step stream (the tool-agent family shares it).
STEP_SCHEMA = "hud.step.v1"
ROBOT_STEP_SCHEMA = "hud.robot.step.v1"

StepSource: TypeAlias = Literal["user", "agent", "tool", "task", "subagent", "system"]
RobotStepSource: TypeAlias = Literal["observation", "inference", "video_segment"]


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

    Agent families subclass it with their own payload fields (such as
    ``hud.agents.types.AgentStep``) and, for a new wire schema, their own ``schema_tag``.
    """

    # The platform's serializer registry dispatches on this tag.
    schema_tag: ClassVar[str] = STEP_SCHEMA

    # Sequential position in the trace, assigned by ``Trace`` (1-based).
    step_id: int | None = None
    source: StepSource

    messages: list[PromptMessage] = Field(default_factory=list[PromptMessage])
    task_call: TaskCall | None = None

    error: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    extra: JsonObject = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    def emit(self, *, trace_id: str | None = None) -> None:
        """Export this step as a span with its schema; a no-op without a trace context.

        Pass ``trace_id`` when emitting outside the rollout's context, such as from a thread.
        """

        task_run_id = trace_id or get_current_trace_id()
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

# Why the rollout stopped; anything but "done" means a limit cut it off.
StopReason: TypeAlias = Literal["done", "max_steps", "length", "timeout", "malformed_tool_call"]

# The configurable subset of stop reasons (``ToolAgentConfig.stop_on``): policy
# conditions the loop may either stop on or answer with an error result.
StopCondition: TypeAlias = Literal["length", "malformed_tool_call"]


class Grade(BaseModel):
    """The environment's verdict: ``raw`` is its grade frame, ``content`` its message."""

    reward: float = 0.0
    done: bool = True
    content: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    is_error: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    _evaluation: dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        # Malformed subscores fail here, where the grade is built, not when read.
        evaluation = dict(self.raw)
        if isinstance(subscores := evaluation.get("subscores"), list):
            evaluation["subscores"] = [
                SubScore.model_validate(subscore).to_summary() for subscore in subscores
            ]
        self._evaluation = evaluation

    @property
    def evaluation(self) -> dict[str, Any]:
        """A persistence-safe view of ``raw``: subscores reduced to their summaries."""
        return dict(self._evaluation)

    @classmethod
    def ungraded(cls) -> Grade:
        """The grade of a trace that has not been graded."""
        return cls(is_error=True, content="not graded")


class Trace(BaseModel):
    """The record of one evaluation: its ordered steps, the agent's answer and the grade.

    Summaries such as ``error`` derive from the steps through ``final`` and ``collect``.
    """

    steps: list[SerializeAsAny[Step]] = Field(default_factory=list[Step])

    status: TraceStatus | None = None
    stop_reason: StopReason | None = None
    content: str | None = Field(default=None)
    grade: Grade = Field(default_factory=Grade.ungraded)

    # Metadata with no structured home, such as provider session info.
    extra: JsonObject = Field(default_factory=dict)

    trace_id: str | None = Field(default=None)
    job_id: str | None = None
    group_id: str | None = None
    slug: str | None = None
    # Where the evaluation ran: its control-channel url, local://<env> for an environment
    # in the evaluating process, or hud://trace/<id> when hosted.
    runtime: str | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def failed(cls, error: str) -> Trace:
        """The trace of an evaluation that failed before it started."""
        return cls(status="error", steps=[Step(source="system", error=error)])

    @property
    def reward(self) -> float:
        return self.grade.reward

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.grade.evaluation

    def final(self, get: Callable[[Step], T | None]) -> T | None:
        """The newest step's non-``None`` answer to ``get``; falsy answers count."""
        return next(
            (value for step in reversed(self.steps) if (value := get(step)) is not None),
            None,
        )

    def collect(self, get: Callable[[Step], T | None]) -> list[T]:
        """Every step's non-``None`` answer to ``get``, in step order."""
        return [value for step in self.steps if (value := get(step)) is not None]

    @property
    def is_error(self) -> bool:
        return self.status == "error"

    @property
    def is_truncated(self) -> bool:
        """Whether a limit (step budget, token cap, timeout) cut the rollout off."""
        return self.stop_reason is not None and self.stop_reason != "done"

    @property
    def error(self) -> str | None:
        """The most recent step error."""
        return self.final(lambda step: step.error)

    @property
    def prompt_text(self) -> str | None:
        """The opening user step's text turns, or ``None`` when no task started."""
        user = next((step for step in self.steps if step.source == "user"), None)
        return None if user is None else text_of(user.messages)

    @model_validator(mode="after")
    def _number_steps(self) -> Trace:
        for index, step in enumerate(self.steps, start=1):
            step.step_id = index
        return self

    def record(self, step: Step) -> None:
        """Number, append and stream one step; callers stamp its ``started_at``."""
        step.step_id = len(self.steps) + 1
        if step.ended_at is None:
            step.ended_at = now_iso()
        self.steps.append(step)
        step.emit()

    def __len__(self) -> int:
        return len(self.steps)
