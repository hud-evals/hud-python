"""Agent configs, importable without provider SDKs, and the step payloads agents record."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, NotRequired, Self, cast
from uuid import uuid4

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
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
    ROBOT_STEP_SCHEMA,
    ContentBlock,
    ImageContent,
    RobotStepSource,
    Step,
    StepSource,
    StopCondition,
    TextContent,
    ToolCall,
    ToolResult,
    Trace,
)
from hud.utils.serialization import json_safe_value

_model_alias = AliasChoices("model", "checkpoint_name")

ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]
CodexReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


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
    reasoning: Any = None  # openai Reasoning
    tool_choice: Any = None  # openai ToolChoice
    text: Any = None  # {"verbosity": "low"|"medium"|"high"}
    truncation: Literal["auto", "disabled"] | None = None
    parallel_tool_calls: bool | None = None
    prompt_cache_key: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=64,
    )


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


class AgentStep(Step):
    """The model's turn: one LLM call."""

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


class ToolStep(Step):
    """One tool call and its result.

    A failed call is ``result.isError``; the step's ``error`` is for harness faults.
    """

    source: StepSource = "tool"
    call: ToolCall | None = None
    result: ToolResult | None = None


class SubagentStep(Step):
    """A nested rollout's trace, embedded whole; its steps stream under their own trace id."""

    source: StepSource = "subagent"
    subagent: Trace


class StateFeature(BaseModel):
    """An observation feature group's values and, when the contract has them, their labels."""

    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(default_factory=list[str])
    values: list[float] = Field(default_factory=list[float])


class ObservationStep(Step):
    """What the policy saw at one control tick; ``tick`` pairs it with its ``InferenceStep``."""

    schema_tag: ClassVar[str] = ROBOT_STEP_SCHEMA
    source: RobotStepSource = "observation"

    tick: int = 0
    # TODO: note - this reuses the MCP-native ImageContent type
    images: dict[str, ImageContent] = Field(default_factory=dict[str, ImageContent])
    state: dict[str, StateFeature] = Field(default_factory=dict[str, StateFeature])

    @classmethod
    def from_obs(
        cls,
        obs: dict[str, Any],
        *,
        tick: int = 0,
        obs_space: dict[str, Any] | None = None,
    ) -> ObservationStep:
        """Build a step's ``state`` from a raw ``robot`` obs, labelled by ``obs_space``."""
        obs_space = obs_space or {}
        state: dict[str, StateFeature] = {}
        for name, arr in obs.get("data", {}).items():
            if arr.ndim >= 2:
                continue  # camera frames travel as video, not per-tick images
            vec = arr.tolist()
            # Contract features keyed by this data key (as a dot segment) label the flat vector:
            # ordered slices that tile it become one group each (libero_pro), or one feature named
            # exactly like the key labels it whole (libero_ee_del). Otherwise it stays unlabelled.
            slices: list[tuple[int, int, str, list[str]]] = []
            direct: list[str] | None = None
            for feature_key, feature in obs_space.items():
                if name not in feature_key.split(".") or not isinstance(feature, dict):
                    continue
                feature_meta = cast("dict[str, Any]", feature)
                raw_names = feature_meta.get("names")
                labels = (
                    [str(n) for n in cast("list[Any]", raw_names)]
                    if isinstance(raw_names, list)
                    else []
                )
                order = feature_meta.get("order")
                if order is not None:
                    bounds = str(order).split("-")
                    slices.append(
                        (
                            int(bounds[0]),
                            int(bounds[-1]),
                            feature_key.split(".")[-1],
                            labels,
                        )
                    )
                elif feature_key.split(".")[-1] == name and len(labels) == len(vec):
                    direct = labels
            slices.sort()
            covered = [i for start, end, _, _ in slices for i in range(start, end + 1)]
            if covered == list(range(len(vec))):
                for start, end, key, labels in slices:
                    values = vec[start : end + 1]
                    state[key] = StateFeature(
                        names=labels if len(labels) == len(values) else [],
                        values=values,
                    )
            elif direct is not None:
                state[name] = StateFeature(names=direct, values=vec)
            else:
                state[name] = StateFeature(values=vec)
        return cls(tick=tick, state=state)


class InferenceStep(Step):
    """The ``[T, A]`` action chunk the policy executed from one control tick.

    A single action is a length-1 chunk; ``tick`` matches the paired observation.
    """

    schema_tag: ClassVar[str] = ROBOT_STEP_SCHEMA
    source: RobotStepSource = "inference"

    tick: int = 0  # start of inference
    # end_tick: int = 0 # end of inference - future implementation

    chunk: list[list[float]] = Field(default_factory=list[list[float]])
    chunk_length: int = 1
    # Per-action-dimension labels from the env contract, for the viewer's action plots.
    names: list[str] = Field(default_factory=list[str])


class VideoSegmentStep(Step):
    """One CMAF (fragmented MP4) fragment of a camera's H.264 stream."""

    schema_tag: ClassVar[str] = ROBOT_STEP_SCHEMA
    source: RobotStepSource = "video_segment"

    camera: str = ""
    # Position in the camera's stream; ``index`` 0 is the init segment.
    index: int = 0
    fps: int = 10
    # A ``video`` content block (base64 MP4), which ingest offloads to S3 like an image.
    segment: dict[str, Any] = Field(default_factory=dict[str, Any])


class ContentResult(BaseModel):
    """A custom MCP tool's common text-plus-image return."""

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
