"""Agent configuration and the tool-agent family's step payloads.

Config classes are defined here separately from agent implementations
to allow importing them without requiring SDK dependencies (anthropic, google-genai).

The trajectory section layers the tool-agent family on the core contract:
:mod:`hud.types` owns the skeleton (ordering, timing, error, span
transport); this module adds what an LLM tool-use loop produces, flat on
that skeleton — the model's turn (:class:`AgentStep`), the tool round-trip
(:class:`ToolStep` pairing an ``MCPToolCall`` with its ``MCPToolResult``),
nested rollouts (:class:`SubagentStep`), and the token-level training and
accounting vocabulary (:class:`Sample`, :class:`Usage`). All ship under
the core ``hud.step.v1`` schema — the platform's serializer for that
schema understands this family's payload.
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.types import ContentBlock, ImageContent, TextContent
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
)

from hud.agents.tools.hosted import HostedTool
from hud.types import MCPToolCall, MCPToolResult, Step, StepSource, Trace
from hud.utils.serialization import json_safe_value

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
# Trajectory (tool-agent family step payloads)
# -----------------------------------------------------------------------------


class Citation(BaseModel):
    """Normalized citation from any provider.

    Unifies OpenAI ``url_citation``/``file_citation`` annotations, Claude ``cite``
    blocks, and Gemini grounding into a single shape: a span of agent output linked
    to its source. The ``type`` field preserves the provider-specific category.
    A reply annotation, not a grading input: provider agents attach these to the
    turn (``AgentStep.citations``), where chat surfaces and the platform read
    them — e.g. the final reply's citations are
    ``trace.final(lambda s: s.citations if isinstance(s, AgentStep) else None)``.
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


class Sample(BaseModel):
    """One model generation in a rollout: tokens conditioned on + tokens produced.

    Token-level data for RL training (Tinker-shaped). ``output_logprobs`` are the
    per-output-token logprobs under the *sampling* policy (q). Populated only when
    the model backend is trainable (returns token ids + logprobs); closed/eval-only
    backends leave it empty.
    """

    prompt_token_ids: list[int] = Field(default_factory=list[int])
    output_token_ids: list[int] = Field(default_factory=list[int])
    output_logprobs: list[float] = Field(default_factory=list[float])


class Usage(BaseModel):
    """Normalized per-step usage accounting.

    Provider responses report usage under different keys; this is the canonical
    accounting shape (token-level training data lives in ``Sample``).
    ``llm_call_count`` is for aggregate steps that wrap several calls.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    llm_call_count: int | None = None


class AgentStep(Step):
    """The model's turn: one LLM call, flat on the step skeleton.

    Provider agents return one from ``get_response()`` and the loop records
    it directly — timing lands on ``started_at``/``ended_at`` and failures
    on ``error``, the skeleton's own channels. ``usage`` is the turn's
    normalized accounting (aggregate turns wrapping several calls report
    ``llm_call_count``); ``sample`` carries this turn's token-level
    training data iff the model backend is trainable.
    """

    source: StepSource = "agent"

    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[MCPToolCall] = Field(default_factory=list[MCPToolCall])
    #: No further tool calls expected — the loop's stop signal.
    done: bool = False
    finish_reason: str | None = None
    refusal: str | None = None
    citations: list[Citation] = Field(default_factory=list[Citation])
    raw: Any | None = None

    model: str | None = None
    usage: Usage | None = None
    sample: Sample | None = None

    @field_serializer("raw", when_used="json")
    def _serialize_raw(self, raw: Any | None) -> Any:
        return json_safe_value(raw)


class ToolStep(Step):
    """One tool round-trip: the originating call paired with its result.

    Error-ness of the call is data on the ``result`` (``isError``), not a
    step failure; ``error`` stays for harness-level faults.
    """

    source: StepSource = "tool"
    call: MCPToolCall | None = None
    result: MCPToolResult | None = None


class SubagentStep(Step):
    """A nested rollout (e.g. an ``AgentTool`` invocation), embedded whole.

    The sub-rollout's own steps stream under its own trace id; this step is
    the enclosing trace's record of the invocation.
    """

    source: StepSource = "subagent"
    subagent: Trace


class ContentResult(BaseModel):
    """Ergonomic builder for a custom MCP tool's ``list[ContentBlock]`` return.

    A ``@server.tool`` returns content blocks; this assembles the common
    text (+ optional image) case in one line so vision tools — games,
    computer-use, browsers — don't hand-roll the same block list::

        from hud.agents.types import ContentResult

        @server.tool
        async def look() -> list[ContentBlock]:
            return ContentResult(output=status, base64_image=png_b64).to_content_blocks()
    """

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None

    def to_content_blocks(self) -> list[ContentBlock]:
        """Text block(s) for ``output``/``error``, plus an image for ``base64_image``."""
        blocks: list[ContentBlock] = []
        if self.output:
            blocks.append(TextContent(type="text", text=self.output))
        if self.error:
            blocks.append(TextContent(type="text", text=self.error))
        if self.base64_image:
            mime = "image/jpeg" if self.base64_image.startswith("/9j/") else "image/png"
            blocks.append(ImageContent(type="image", data=self.base64_image, mimeType=mime))
        return blocks
