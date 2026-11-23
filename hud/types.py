"""Universal SDK shapes, including the trajectory contract."""

from __future__ import annotations

import contextlib
import json
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeAlias

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializeAsAny,
    model_validator,
)
from typing_extensions import TypeVar

from hud.graders.base import Grade
from hud.telemetry.context import get_current_trace_id
from hud.utils.deprecation import moved_to, resolve_retired, warn_deprecated
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
    PI = "pi"

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
            case AgentType.PI:
                from hud.agents.pi.agent import PiAgent

                return PiAgent

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
            PiConfig,
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
            case AgentType.PI:
                return PiConfig

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


# Schema tag of the core step stream (the tool-agent family shares it).
STEP_SCHEMA = "hud.step.v1"

StepSource: TypeAlias = Literal["user", "agent", "tool", "task", "subagent", "system"]
SourceT_co = TypeVar("SourceT_co", covariant=True, default=StepSource)


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
    A family parametrizes ``Step`` with its own ``source`` values; a bare ``Step`` takes
    ``StepSource``.
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
        """Export this step under ``trace_id``, else the current trace; deprecated."""
        # hud.eval.run imports this module, so the deprecated path imports it late.
        from hud.eval.run import export_step

        warn_deprecated("`Step.emit`", "use `hud.eval.run.export_step(trace_id, step)`")
        trace_id = trace_id or get_current_trace_id()
        if trace_id:
            export_step(trace_id, self)


TraceStatus: TypeAlias = Literal["completed", "error", "cancelled"]

# Why the rollout stopped; anything but "done" means a limit cut it off.
StopReason: TypeAlias = Literal["done", "max_steps", "length", "timeout", "malformed_tool_call"]

# The configurable subset of stop reasons (``ToolAgentConfig.stop_on``): policy
# conditions the loop may either stop on or answer with an error result.
StopCondition: TypeAlias = Literal["length", "malformed_tool_call"]


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
        """``grade.to_summary()``; deprecated."""
        warn_deprecated("`Trace.evaluation`", "use `trace.grade.to_summary()`")
        return self.grade.to_summary()

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
        """Whether a limit (step budget, token cap, timeout) cut the rollout off; deprecated."""
        warn_deprecated(
            "`Trace.is_truncated`",
            "read `trace.stop_reason` (anything but `None` or `'done'` means a limit cut it off)",
        )
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
        """Number and append one step; callers stamp its ``started_at``."""
        step.step_id = len(self.steps) + 1
        if step.ended_at is None:
            step.ended_at = now_iso()
        self.steps.append(step)

    def __len__(self) -> int:
        warn_deprecated("`len(trace)`", "use `len(trace.steps)`")
        return len(self.steps)


_RETIRED = moved_to("hud.graders", "EvaluationResult", "SubScore")


def __getattr__(name: str) -> Any:
    return resolve_retired(__name__, name, _RETIRED)
