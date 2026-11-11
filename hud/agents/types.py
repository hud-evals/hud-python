"""Agent configs and the step payloads agents record."""

from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired, Self, cast
from uuid import uuid4

from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_create_params import ToolChoice
from openai.types.shared_params.reasoning import Reasoning
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_validator,
    with_config,
)
from typing_extensions import TypedDict

from hud.agents.tools.hosted import HostedTool
from hud.capabilities.rfb import (
    PngScreenshotEncoding,
    ScreenshotEncoding,
    WebPScreenshotEncoding,
)
from hud.types import (
    Step,
    StepSource,
    StopCondition,
    ToolCall,
    ToolResult,
    Trace,
)
from hud.utils.serialization import json_safe_value

# Agent configs written before the field was named ``model`` call it ``checkpoint_name``;
# task rows store them in ``agent_config``.
_model_alias = AliasChoices("model", "checkpoint_name")

ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]
CodexReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
# A name a shell assignment accepts.
EnvironmentName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class AgentConfig(BaseModel):
    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    model_name: str = "Agent"
    model: str = Field(default="unknown", validation_alias=_model_alias)
    gateway: bool = False


class ToolAgentConfig(AgentConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auto_respond: bool = False
    max_steps: int = 10
    tool_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    system_prompt: str | None = None
    citations_enabled: bool = False
    # Turn conditions that end the rollout rather than leaving the model to recover.
    stop_on: set[StopCondition] = Field(default_factory=set[StopCondition])
    hosted_tools: list[HostedTool[object]] = Field(
        default_factory=list[HostedTool[object]], exclude=True
    )
    screenshot_encoding: ScreenshotEncoding = Field(default_factory=PngScreenshotEncoding)
    model_client: Any = Field(default=None, exclude=True)


@with_config(ConfigDict(extra="forbid"))
class ClaudeAdaptiveThinkingConfig(TypedDict):
    type: Literal["adaptive"]
    display: NotRequired[Literal["summarized", "omitted"]]


@with_config(ConfigDict(extra="forbid"))
class ClaudeEnabledThinkingConfig(TypedDict):
    type: Literal["enabled"]
    budget_tokens: Annotated[int, Field(ge=1024)]
    display: NotRequired[Literal["summarized", "omitted"]]


@with_config(ConfigDict(extra="forbid"))
class ClaudeDisabledThinkingConfig(TypedDict):
    type: Literal["disabled"]


ClaudeThinkingConfig = (
    ClaudeAdaptiveThinkingConfig | ClaudeEnabledThinkingConfig | ClaudeDisabledThinkingConfig
)


class ClaudeConfig(ToolAgentConfig):
    model_name: str = "Claude"
    model: str = Field(default="claude-sonnet-4-6", validation_alias=_model_alias)
    tool_timeout_seconds: float | None = Field(default=120, gt=0, allow_inf_nan=False)
    max_tokens: int = 16384
    use_computer_beta: bool = True
    screenshot_encoding: ScreenshotEncoding = Field(default_factory=WebPScreenshotEncoding)
    thinking: ClaudeThinkingConfig | None = None
    # Sent as ``output_config.effort``; unset leaves the model's own default.
    reasoning_effort: ClaudeEffort | None = None
    max_tool_result_images: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_thinking(self) -> Self:
        if (
            self.thinking is not None
            and self.thinking["type"] == "enabled"
            and self.thinking["budget_tokens"] >= self.max_tokens
        ):
            raise ValueError("enabled thinking budget_tokens must be less than max_tokens")
        if self.max_tool_result_images is not None and (
            self.thinking is None or self.thinking["type"] == "disabled"
        ):
            raise ValueError(
                "max_tool_result_images requires explicit adaptive or enabled thinking"
            )
        return self


class GeminiConfig(ToolAgentConfig):
    model_name: str = "Gemini"
    model: str = Field(default="gemini-3.1-pro-preview", validation_alias=_model_alias)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_output_tokens: int | None = None
    excluded_predefined_functions: list[str] = Field(default_factory=list)
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    include_thoughts: bool = True
    max_tool_result_chars: int = 64 * 1024


class OpenAIConfig(ToolAgentConfig):
    model_name: str = "OpenAI"
    model: str = Field(default="gpt-5.6", validation_alias=_model_alias)
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning: Reasoning | None = None
    tool_choice: ToolChoice | None = None
    text: ResponseTextConfigParam | None = None
    truncation: Literal["auto", "disabled"] | None = None
    parallel_tool_calls: bool | None = None
    prompt_cache_key: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=64,
    )

    @field_validator("reasoning", "tool_choice", "text", mode="wrap")
    @classmethod
    def _declared_keys_only(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """Raise on a key the OpenAI SDK's type does not declare, which validation would drop."""
        validated = handler(value)
        if dropped := _dropped_keys(value, validated):
            raise ValueError(f"unknown keys for the installed openai SDK: {', '.join(dropped)}")
        return validated


def _dropped_keys(raw: Any, kept: Any, path: str = "") -> list[str]:
    if not isinstance(raw, dict) or not isinstance(kept, dict):
        return []
    raw_dict, kept_dict = cast("dict[str, Any]", raw), cast("dict[str, Any]", kept)
    return [
        dropped
        for key, value in raw_dict.items()
        for dropped in (
            [f"{path}{key}"]
            if key not in kept_dict
            else _dropped_keys(value, kept_dict[key], f"{path}{key}.")
        )
    ]


class OpenAIChatConfig(ToolAgentConfig):
    model_name: str = "OpenAI Chat"
    model: str = Field(default="gpt-5.4-mini", validation_alias=_model_alias)
    checkpoint: str | None = Field(
        default=None,
        description="Specific checkpoint name for inference routing. "
        "When set, the HUD gateway routes to this exact checkpoint rather than "
        "the model's current active checkpoint. Passed as 'checkpoint' in the "
        "request body's extra_body.",
    )
    api_key: str | None = Field(default=None, exclude=True)
    base_url: str | None = Field(default=None, exclude=True)
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)


class ClaudeCLIConfig(AgentConfig):
    """``max_steps`` is the CLI's ``--max-turns``; values <= 0 leave it unlimited."""

    system_prompt: str | None = None
    model_name: str = "Claude CLI"
    model: str = Field(default="claude-sonnet-5", validation_alias=_model_alias)
    permission_mode: str = "bypassPermissions"
    max_steps: int = -1
    # Maps to the CLI's ``--effort``; unset leaves the CLI's own default.
    reasoning_effort: ClaudeEffort | None = None
    # Extra variables for the CLI's environment, such as its model tiers. They reach
    # the CLI as given, inside the workspace, and travel with the serialized config.
    # No key may name a variable the agent sets; names compare case-insensitively.
    env: dict[EnvironmentName, str] = Field(default_factory=dict)
    screenshot_encoding: ScreenshotEncoding = Field(default_factory=WebPScreenshotEncoding)
    allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "Read",
            "Write",
            "Edit",
            "Bash",
            "Glob",
            "Grep",
            "WebSearch",
            "WebFetch",
        ],
    )
    disallowed_tools: list[str] = Field(default_factory=list)


class CodexCLIConfig(AgentConfig):
    """``sandbox`` does not apply under an inference connection; the workspace isolates it.

    Without injected credentials, the environment's own Codex login applies.
    """

    model_name: str = "Codex CLI"
    model: str = Field(default="gpt-5.6-sol", validation_alias=_model_alias)
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "workspace-write"
    # Maps to ``model_reasoning_effort``; unset leaves the CLI's own default.
    reasoning_effort: CodexReasoningEffort | None = None


class Citation(BaseModel):
    """A span of agent output linked to its source, normalized across providers."""

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
        default=None,
        description="Start character index in the agent's output text",
    )
    end_index: int | None = Field(
        default=None,
        description="End character index in the agent's output text",
    )


class Sample(BaseModel):
    """One generation's prompt and output tokens, recorded only by trainable backends.

    ``output_logprobs`` are under the sampling policy.
    """

    prompt_token_ids: list[int] = Field(default_factory=list[int])
    # Serialized ``ModelInput`` chunks of a multimodal prompt; when set, they are the prompt
    # and ``prompt_token_ids`` is empty.
    prompt_chunks: list[dict[str, Any]] | None = None
    output_token_ids: list[int] = Field(default_factory=list[int])
    output_logprobs: list[float] = Field(default_factory=list[float])


class Usage(BaseModel):
    """Per-step token and cost accounting, normalized across providers."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None
    # Set on aggregate steps that wrap several calls.
    llm_call_count: int | None = None


class AgentStep(Step[StepSource]):
    """One model output: a ``ToolAgent``'s LLM call, or one streamed item from a CLI agent."""

    source: StepSource = "agent"

    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    # No further tool calls expected — the loop's stop signal.
    done: bool = False
    finish_reason: str | None = None
    stop_reason: StopCondition | None = None
    refusal: str | None = None
    citations: list[Citation] = Field(default_factory=list[Citation])
    raw: Any | None = None

    model: str | None = None
    usage: Usage | None = None
    sample: Sample | None = None

    @field_serializer("raw", when_used="json")
    def _serialize_raw(self, raw: Any | None) -> Any:
        return json_safe_value(raw)


class ToolStep(Step[StepSource]):
    """One tool call and its result.

    A failed call is ``result.isError``; the step's ``error`` is for harness faults.
    """

    source: StepSource = "tool"
    call: ToolCall | None = None
    result: ToolResult | None = None


class SubagentStep(Step[StepSource]):
    """A nested rollout's trace, embedded whole; its steps stream under their own trace id."""

    source: StepSource = "subagent"
    subagent: Trace
