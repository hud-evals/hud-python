"""Agent configuration + result types.

Config classes are defined here separately from agent implementations
to allow importing them without requiring SDK dependencies (anthropic, google-genai).
This module also holds the agent-facing result/answer types (``Citation``,
``AgentAnswer``, ``ScenarioResult``/``EvaluationResult``, ``ContentResult``,
``SubScore``, ``ToolError``) — the serializable shapes agents and scenarios exchange.
"""

from __future__ import annotations

import warnings
from typing import Any, Generic, Literal, TypeVar

from mcp.types import ContentBlock, ImageContent, TextContent
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from hud.agents.tools.hosted import HostedTool
from hud.types import Trace

T = TypeVar("T")

# Alias to accept both 'model' and 'checkpoint_name' (backwards compat)
_model_alias = AliasChoices("model", "checkpoint_name")


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auto_respond: bool = False
    max_steps: int = 10
    system_prompt: str | None = None
    citations_enabled: bool = False
    hosted_tools: list[HostedTool[object]] = Field(default_factory=list[HostedTool[object]])

    model_name: str = "Agent"
    model: str = Field(default="unknown", validation_alias=_model_alias)
    #: Provider client (AsyncAnthropic, AsyncOpenAI, genai.Client, ...). When unset,
    #: agents resolve one from settings (HUD gateway or provider API key).
    model_client: Any = None


# -----------------------------------------------------------------------------
# Claude
# -----------------------------------------------------------------------------


class ClaudeConfig(AgentConfig):
    model_name: str = "Claude"
    model: str = Field(default="claude-sonnet-4-6", validation_alias=_model_alias)
    max_tokens: int = 16384
    use_computer_beta: bool = True


# -----------------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------------


class GeminiConfig(AgentConfig):
    """Configuration for GeminiAgent."""

    model_name: str = "Gemini"
    model: str = Field(default="gemini-3-pro-preview", validation_alias=_model_alias)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    excluded_predefined_functions: list[str] = Field(default_factory=list)
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    include_thoughts: bool = True


# -----------------------------------------------------------------------------
# OpenAI
# -----------------------------------------------------------------------------


class OpenAIConfig(AgentConfig):
    """Configuration for OpenAIAgent."""

    model_name: str = "OpenAI"
    model: str = Field(default="gpt-5.4", validation_alias=_model_alias)
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning: Any = None  # openai Reasoning
    tool_choice: Any = None  # openai ToolChoice
    text: Any = None  # {"verbosity": "low"|"medium"|"high"}
    truncation: Literal["auto", "disabled"] | None = None
    parallel_tool_calls: bool | None = None


class OpenAIChatConfig(AgentConfig):
    """Configuration for OpenAIChatAgent."""

    model_name: str = "OpenAI Chat"
    model: str = Field(default="gpt-5-mini", validation_alias=_model_alias)
    checkpoint: str | None = Field(
        default=None,
        description="Specific checkpoint name for inference routing. "
        "When set, the HUD gateway routes to this exact checkpoint rather than "
        "the model's current active checkpoint. Passed as 'checkpoint' in the "
        "request body's extra_body.",
    )
    api_key: str | None = None
    base_url: str | None = None
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Claude Code (CLI over SSH)
# -----------------------------------------------------------------------------


class ClaudeSDKConfig(AgentConfig):
    """Configuration for ClaudeSDKAgent (runs the ``claude`` CLI over SSH).

    ``system_prompt`` is inherited from ``AgentConfig``. ``max_steps`` maps to the
    CLI's ``--max-turns``; values <= 0 leave the turn budget to the CLI (unlimited).
    """

    model_name: str = "Claude Code"
    model: str = Field(default="claude-sonnet-4-5", validation_alias=_model_alias)
    permission_mode: str = "bypassPermissions"
    max_steps: int = -1
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
    )


# -----------------------------------------------------------------------------
# Browser Use
# -----------------------------------------------------------------------------


class BrowserUseConfig(AgentConfig):
    """Configuration for BrowserUseAgent.

    Lives here (not in the agent module) so it can be imported and serialized
    without the optional ``browser-use`` dependency installed. The ``auto_respond``
    / ``system_prompt`` / ``hosted_tools`` fields from ``AgentConfig`` do not apply
    — browser-use runs its own agent loop.
    """

    model_name: str = "Browser Use"
    model: str = Field(default="claude-sonnet-4-5", validation_alias=_model_alias)
    api_key: str | None = None
    base_url: str | None = None
    max_steps: int = 25


# -----------------------------------------------------------------------------
# Result / answer types (exchanged between agents, tools, and scenarios)
# -----------------------------------------------------------------------------


class SubScore(BaseModel):
    """Individual subscore for debugging and transparency.

    SubScores allow breaking down the final reward into component parts,
    making it easier to understand what contributed to the evaluation.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of this subscore component")
    weight: float = Field(
        default=1.0,
        description="Weight of this subscore (for weighted average). "
        "Negative weights represent penalties.",
    )
    value: float = Field(..., ge=0.0, le=1.0, description="Value of this subscore, 0.0 to 1.0")
    metadata: dict[str, Any] | None = Field(default=None, exclude=True)

    @property
    def score(self) -> float:
        """Alias for value. Deprecated — use .value instead."""
        return self.value


class ScenarioResult(BaseModel):
    """Result from a scenario's final phase.

    In eval mode, populate reward and subscores for scoring.
    In production, use content and info for diagnostics and stats.
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

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _check_subscores(self) -> ScenarioResult:
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
    def from_float(cls, value: float) -> ScenarioResult:
        """Create a ScenarioResult from a simple float reward."""
        return cls(reward=value, done=True)


EvaluationResult = ScenarioResult


class ContentResult(BaseModel):
    """Represents the intermediate result of a tool execution.

    Often useful for tools that need to return multiple types of content.
    """

    output: str | None = Field(default=None, description="Output text")
    error: str | None = Field(default=None, description="Error message")
    base64_image: str | None = Field(default=None, description="Base64-encoded image")
    system: str | None = Field(default=None, description="System message")
    url: str | None = Field(default=None, description="Current page URL (for browser automation)")

    def __add__(self, other: ContentResult) -> ContentResult:
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ) -> str | None:
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ContentResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
            url=combine_fields(self.url, other.url, False),
        )

    def to_text_blocks(self) -> list[TextContent]:
        """Convert text-only content to TextContent blocks."""
        blocks: list[TextContent] = []
        if self.output:
            blocks.append(TextContent(text=self.output, type="text"))
        if self.error:
            blocks.append(TextContent(text=self.error, type="text"))
        if self.url:
            blocks.append(TextContent(text=f"__URL__:{self.url}", type="text"))
        return blocks

    def to_content_blocks(self) -> list[ContentBlock]:
        """Convert to content blocks including images."""
        blocks: list[ContentBlock] = list(self.to_text_blocks())
        if self.base64_image:
            mime = "image/jpeg" if self.base64_image.startswith("/9j/") else "image/png"
            blocks.append(ImageContent(data=self.base64_image, mimeType=mime, type="image"))
        return blocks


class Citation(BaseModel):
    """Normalized citation from any provider.

    Unifies OpenAI ``url_citation``/``file_citation`` annotations, Claude ``cite``
    blocks, and Gemini grounding into a single shape: a span of agent output linked
    to its source. The ``type`` field preserves the provider-specific category.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        default="citation",
        description="Citation kind: 'url_citation', 'file_citation', "
        "'document_citation', 'grounding', or generic 'citation'",
    )
    text: str = Field(default="", description="The cited passage or annotated text span")
    source: str = Field(default="", description="URL, file ID, or document identifier")
    title: str | None = Field(default=None, description="Title of the source")
    start_index: int | None = Field(
        default=None, description="Start character index in the agent's output text"
    )
    end_index: int | None = Field(
        default=None, description="End character index in the agent's output text"
    )
    provider_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw provider-specific data for advanced use",
    )


class AgentAnswer(BaseModel, Generic[T]):
    """Wrapper holding an agent's structured answer alongside response metadata.

    When a scenario specifies ``returns=SomeModel``, the answer received by the
    scenario's evaluate phase is an ``AgentAnswer[SomeModel]``: a parsed ``content``,
    the original ``raw`` string, normalized ``citations``, and optional ``trace``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: T = Field(description="The parsed structured answer")
    raw: str = Field(default="", description="Original answer string before parsing")
    citations: list[Citation] = Field(default_factory=list[Citation])
    trace: Trace | None = Field(
        default=None,
        description="Full conversation transcript (multi-turn). "
        "Populated by AgentService for multi-turn sessions.",
    )


class ToolError(Exception):
    """An error raised by a tool."""
