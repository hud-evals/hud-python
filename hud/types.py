"""Universal SDK shapes, including the trajectory contract."""

from __future__ import annotations

import contextlib
import json
import uuid
import warnings
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeAlias, cast

from pydantic import (
    AliasChoices,
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    SerializationInfo,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)
from rich.markup import escape
from typing_extensions import TypeVar

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
from hud.utils.hud_console import hud_console
from hud.utils.serialization import JsonObject
from hud.utils.time import now_iso

if TYPE_CHECKING:
    from collections.abc import Callable

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
        from hud.agents.cli import CLIAgent

        return issubclass(self.cls, CLIAgent)

    @property
    def cls(self) -> type[Agent]:
        match self:
            case AgentType.CLAUDE:
                from hud.agents.claude.agent import ClaudeAgent

                return ClaudeAgent
            case AgentType.CLAUDE_CLI:
                from hud.agents.claude.cli.agent import ClaudeCLIAgent

                return ClaudeCLIAgent
            case AgentType.CODEX_CLI:
                from hud.agents.codex.agent import CodexCLIAgent

                return CodexCLIAgent
            case AgentType.OPENAI:
                from hud.agents.openai.agent import OpenAIAgent

                return OpenAIAgent
            case AgentType.GEMINI:
                from hud.agents.gemini.agent import GeminiAgent

                return GeminiAgent
            case AgentType.OPENAI_COMPATIBLE:
                from hud.agents.openai_compatible.agent import OpenAIChatAgent

                return OpenAIChatAgent

    @property
    def config_cls(self) -> type[AgentConfig]:
        """The agent's config class, without importing the agent."""
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


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageContent(BaseModel):
    type: Literal["image"]
    data: str
    mimeType: str


class AudioContent(BaseModel):
    type: Literal["audio"]
    data: str
    mimeType: str


class ResourceLink(BaseModel):
    type: Literal["resource_link"]
    name: str
    title: str | None = None
    uri: AnyUrl
    description: str | None = None
    mimeType: str | None = None
    size: int | None = None


class TextResourceContents(BaseModel):
    uri: AnyUrl
    mimeType: str | None = None
    text: str


class BlobResourceContents(BaseModel):
    uri: AnyUrl
    mimeType: str | None = None
    blob: str


class EmbeddedResource(BaseModel):
    type: Literal["resource"]
    resource: TextResourceContents | BlobResourceContents


ContentBlock: TypeAlias = (
    TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource
)


class PromptMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: ContentBlock


class Tool(BaseModel):
    """A tool an environment offers the agent."""

    name: str
    description: str | None = None
    inputSchema: dict[str, Any]


class ToolCall(BaseModel):
    name: str
    # A str is the provider's raw arguments that didn't parse as JSON; it is never executed.
    arguments: dict[str, Any] | str | None = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    annotation: str | None = None
    # The provider's own name for the tool, when it differs from ``name``.
    provider_name: str | None = None
    # Whether the call's text and PDF results go back to Claude as citable documents.
    citations_enabled: bool = False

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
        s = hud_console.format_tool_call(self.name, self.arguments)
        if self.annotation:
            s += f"  [bright_black]# {escape(self.annotation)}[/bright_black]"
        return s


class ToolResult(BaseModel):
    content: list[ContentBlock]
    structuredContent: dict[str, Any] | None = None
    isError: bool = False
    call_id: str | None = None

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
        content_summary = self._get_content_summary()
        return hud_console.format_tool_result(content_summary, self.isError)


# Schema tag of the core step stream (the tool-agent family shares it).
STEP_SCHEMA = "hud.step.v1"
ROBOT_STEP_SCHEMA = "hud.robot.step.v1"

StepSource: TypeAlias = Literal["user", "agent", "tool", "task", "subagent", "system"]
RobotStepSource: TypeAlias = Literal["observation", "inference", "video_segment"]
SourceT_co = TypeVar("SourceT_co", covariant=True, default=StepSource | RobotStepSource)


class TaskCall(BaseModel):
    """The task-lifecycle RPC a ``task`` step records.

    ``setup`` is ``tasks.start`` (result carries the opening prompt payload);
    ``evaluate`` is ``tasks.grade`` (result carries the evaluation dict).
    """

    phase: Literal["setup", "evaluate"]
    name: str
    arguments: JsonValue = None
    result: JsonValue = None


class Step(BaseModel, Generic[SourceT_co]):
    """One ordered interaction unit in a task run — the shared skeleton.

    Agent families subclass it with their own payload fields (such as
    ``hud.agents.types.AgentStep``) and, for a new wire schema, their own ``schema_tag``.
    A family parametrizes ``Step`` with its own ``source`` values
    (``Step[StepSource]``, ``Step[RobotStepSource]``); a bare ``Step`` takes any family's.
    """

    # The platform's serializer registry dispatches on this tag.
    schema_tag: ClassVar[str] = STEP_SCHEMA

    # Sequential position in the trace, assigned by ``Trace`` (1-based).
    step_id: int | None = None
    source: SourceT_co

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
# Who owns a classified failure: the task's author (``user``), the platform
# running the rollout (``platform``), or the host's provider (``provider``).
FailureSource: TypeAlias = Literal["user", "platform", "provider"]


class ClassifiedError(RuntimeError):
    """An error a rollout reports under a classification.

    ``code`` names the failure for whoever reads the trace (``container_start_failed``,
    ``image_pull_denied``); ``source`` says who owns it. A provider or a runner raises
    one, or a subclass of one, and ``rollout`` records the
    classification on the trace it ends. An unclassified exception ends the run the
    same way without one.
    """

    def __init__(self, message: str, *, code: str, source: FailureSource) -> None:
        super().__init__(message)
        self.code = code
        self.source = source


# The configurable subset of stop reasons (``ToolAgentConfig.stop_on``): policy
# conditions the loop may either stop on or answer with an error result.
StopCondition: TypeAlias = Literal["length", "malformed_tool_call"]


_SUMMARY_SERIALIZATION_CONTEXT = "hud_summary"


class SubScore(BaseModel):
    """One node in the grade breakdown tree.

    A leaf subscore is a single measurement. A parent preserves the subscores
    used to derive its value in ``children``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of this subscore component")
    weight: float = Field(
        default=1.0,
        description="Weight of this subscore (for weighted average). "
        "Negative weights represent penalties.",
    )
    value: float = Field(..., ge=0.0, le=1.0, description="Value of this subscore, 0.0 to 1.0")
    children: list[SubScore] | None = Field(
        default=None,
        description="Input subscores this node was combined from",
    )
    # Environments built before ``info`` was named send it as ``metadata``.
    info: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("info", "metadata"),
        description="Grader-specific details recorded on the evaluation span",
    )

    @model_serializer(mode="wrap")
    def _serialize(
        self,
        handler: SerializerFunctionWrapHandler,
        serialization_info: SerializationInfo,
    ) -> dict[str, Any]:
        data = cast("dict[str, Any]", handler(self))
        if (serialization_info.context or {}).get(_SUMMARY_SERIALIZATION_CONTEXT):
            data.pop("info", None)
        return data

    def to_summary(self) -> dict[str, Any]:
        """Serialize the score breakdown without span-only info."""
        return self.model_dump(
            mode="json",
            context={_SUMMARY_SERIALIZATION_CONTEXT: True},
        )


def absolute_path_below_root(value: str) -> str:
    """Validate a path inside a box: absolute, beneath ``/``, with no ``..`` component."""
    path = PurePosixPath(value)
    if not path.is_absolute() or len(path.parts) == 1 or ".." in path.parts:
        raise ValueError("path must name a location beneath /")
    return str(path)


class CollectCommand(BaseModel):
    """Run ``command`` with ``sh -c`` in a service of the actor's environment."""

    model_config = ConfigDict(extra="forbid")

    service: str = Field(min_length=1)
    command: str = Field(min_length=1)
    timeout_sec: float = Field(gt=0)


class CollectFiles(BaseModel):
    """Copy ``path`` out of a service of the actor's environment into the actor's session.

    It lands at ``into``, a path relative to the session, where a separate verifier finds it.
    """

    model_config = ConfigDict(extra="forbid")

    service: str = Field(min_length=1)
    path: str
    into: str = Field(min_length=1, pattern=r"^[^/]")
    # Skip a path the service does not have instead of failing the run.
    missing_ok: bool = False

    _path_below_root = field_validator("path")(absolute_path_below_root)

    @field_validator("into")
    @classmethod
    def _inside_the_session(cls, value: str) -> str:
        if ".." in value.split("/"):
            raise ValueError("into must stay inside the session")
        return value


class EvaluationResult(BaseModel):
    """The grade a task yields: a reward and an optional ``subscores`` breakdown.

    ``collect`` is what an actor asks the runtime to gather from the other
    services of its environment before its box is torn down, in order, for
    the separate verifier that grades it. Only the runtime can reach those
    services from outside the box.
    """

    reward: float = Field(default=0.0, description="Final score, usually 0.0 to 1.0")
    done: bool = Field(default=True, description="Whether the task/episode is complete")
    content: str | None = Field(default=None, description="Human-readable explanation")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    isError: bool = Field(default=False, description="Whether the evaluation itself failed")
    subscores: list[SubScore] | None = Field(
        default=None,
        description="Optional breakdown of score components for debugging",
    )
    collect: list[CollectCommand | CollectFiles] | None = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _check_subscores(self) -> EvaluationResult:
        if not self.subscores:
            return self
        names = [s.name for s in self.subscores]
        dupes = [n for n in names if names.count(n) > 1]
        if dupes:
            warnings.warn(f"Duplicate subscore names: {set(dupes)}", stacklevel=2)
        pos_weight_sum = sum(s.weight for s in self.subscores if s.weight > 0)
        if abs(pos_weight_sum - 1.0) > 0.01:
            warnings.warn(
                f"Positive subscore weights should sum to ~1.0 (got {pos_weight_sum:.4f}). "
                f"Weights represent proportional contributions to the reward.",
                stacklevel=2,
            )
        weighted_sum = sum(s.value * s.weight for s in self.subscores)
        if abs(weighted_sum - self.reward) > 0.01:
            warnings.warn(
                f"Subscores don't match reward: "
                f"sum(value*weight)={weighted_sum:.4f} but reward={self.reward:.4f}",
                stacklevel=2,
            )
        return self

    @classmethod
    def from_float(cls, value: float) -> EvaluationResult:
        return cls(reward=value, done=True)


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
        evaluation = {key: value for key, value in self.raw.items() if key != "info"}
        if isinstance(subscores := evaluation.get("subscores"), list):
            evaluation["subscores"] = [
                SubScore.model_validate(subscore).to_summary() for subscore in subscores
            ]
        self._evaluation = evaluation

    @property
    def evaluation(self) -> dict[str, Any]:
        """A persistence-safe view of ``raw``: no ``info``, subscores reduced to their summaries.

        ``info`` is span-only, like each subscore's; read it from ``Grade.info``.
        """
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
    # The classification of the error that ended the run, when it carried one.
    error_code: str | None = None
    error_source: FailureSource | None = None
    content: str | None = Field(default=None)
    grade: Grade = Field(default_factory=Grade.ungraded)

    # Metadata with no structured home, such as provider session info.
    extra: JsonObject = Field(default_factory=dict)

    trace_id: str | None = Field(default=None)
    job_id: str | None = None
    group_id: str | None = None
    slug: str | None = None
    # The control-channel url the evaluation ran over, or hud://trace/<id> when hosted.
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
        """The prompt's text turns joined by blank lines (images and resources dropped), or
        ``None`` when no prompt was recorded."""
        user = next((step for step in self.steps if step.source == "user"), None)
        if user is None:
            return None
        return "\n\n".join(
            message.content.text
            for message in user.messages
            if isinstance(message.content, TextContent) and message.content.text
        )

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
