"""Agent configuration types.

Config classes are defined here separately from agent implementations
to allow importing them without requiring SDK dependencies (anthropic, google-genai).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hud.agents.tools.hosted import HostedTool

# Alias to accept both 'model' and 'checkpoint_name' (backwards compat)
_model_alias = AliasChoices("model", "checkpoint_name")


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ctx: Any = None  # EvalContext or Environment
    auto_respond: bool = False
    system_prompt: str | None = None
    hosted_tools: list[HostedTool[object]] = Field(default_factory=list[HostedTool[object]])

    model_name: str = "Agent"
    model: str = Field(default="unknown", validation_alias=_model_alias)


# -----------------------------------------------------------------------------
# Claude
# -----------------------------------------------------------------------------


class ClaudeConfig(AgentConfig):
    model_name: str = "Claude"
    model: str = Field(default="claude-sonnet-4-6", validation_alias=_model_alias)
    model_client: Any = None  # AsyncAnthropic | AsyncAnthropicBedrock
    max_tokens: int = 16384
    use_computer_beta: bool = True
    validate_api_key: bool = True


# -----------------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------------


class GeminiConfig(AgentConfig):
    """Configuration for GeminiAgent."""

    model_name: str = "Gemini"
    model: str = Field(default="gemini-3-pro-preview", validation_alias=_model_alias)
    model_client: Any = None  # AsyncAnthropic | AsyncAnthropicBedrock
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    validate_api_key: bool = True
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
    model_client: Any = None  # AsyncAnthropic | AsyncAnthropicBedrock
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning: Any = None  # openai Reasoning
    tool_choice: Any = None  # openai ToolChoice
    text: Any = None  # {"verbosity": "low"|"medium"|"high"}
    truncation: Literal["auto", "disabled"] | None = None
    parallel_tool_calls: bool | None = None
    validate_api_key: bool = True


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
    openai_client: Any = None  # AsyncOpenAI
    api_key: str | None = None
    base_url: str | None = None
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)
