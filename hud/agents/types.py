"""Agent configs and the step payloads agents record."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    model_validator,
)

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

ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]
CodexReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
# A name a shell assignment accepts.
EnvironmentName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
PiExtensionFile = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+\.(ts|js)$")]


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    model_name: str = "Agent"
    model: str = "unknown"
    gateway: bool = False


class ClaudeCLIConfig(AgentConfig):
    system_prompt: str | None = None
    model_name: str = "Claude CLI"
    model: str = "claude-haiku-5-5"
    permission_mode: str = "bypassPermissions"
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
    model: str = "gpt-6-luna"
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = "workspace-write"
    # Maps to ``model_reasoning_effort``; unset leaves the CLI's own default.
    reasoning_effort: CodexReasoningEffort | None = None


class PiConfig(AgentConfig):
    """Pi calls ``model`` over Chat Completions: through the HUD gateway, or at
    ``OPENAI_BASE_URL`` with ``OPENAI_API_KEY``.

    ``system_prompt`` is appended to Pi's own. ``completion_kwargs`` is merged into every
    request body, with ``extra_body`` entries at the top level. ``model_info`` holds Pi's
    ``models.json`` fields for ``model``, such as ``contextWindow``, ``input`` and ``compat``.
    ``extensions`` maps a file name to the source of a Pi extension loaded into the run.
    Each ``rfb`` screen becomes a ``computer`` MCP tool, so ``model_info["input"]`` must
    include ``"image"`` when the environment has one.
    """

    model_name: str = "Pi"
    model: str = "deepseek/deepseek-v4.1-flash"
    system_prompt: str | None = None
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)
    # The checkpoint of ``model`` the HUD gateway routes to, rather than its active one.
    checkpoint: str | None = None
    model_info: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[PiExtensionFile, str] = Field(default_factory=dict)
    screenshot_encoding: ScreenshotEncoding = Field(default_factory=PngScreenshotEncoding)

    @model_validator(mode="after")
    def _check_token_ids(self) -> Self:
        if "return_token_ids" in self.request_body:
            raise ValueError("pi does not record token ids or logprobs")
        return self

    @property
    def request_body(self) -> dict[str, Any]:
        """The fields Pi merges into every request body."""
        body = {key: value for key, value in self.completion_kwargs.items() if key != "extra_body"}
        body.update(self.completion_kwargs.get("extra_body") or {})
        if self.checkpoint is not None:
            body["checkpoint"] = self.checkpoint
        return body


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
    """One model output: one LLM call, or one streamed item from a CLI agent."""

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
