"""Agent configuration types.

Config classes are defined here separately from agent implementations
to allow importing them without requiring SDK dependencies (anthropic, google-genai).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hud.types import BaseAgentConfig

# Alias to accept both 'model' and 'checkpoint_name' (backwards compat)
_model_alias = AliasChoices("model", "checkpoint_name")


class BaseCreateParams(BaseModel):
    """Runtime parameters for agent creation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ctx: Any = None  # EvalContext or Environment
    auto_respond: bool = False
    verbose: bool = False


# -----------------------------------------------------------------------------
# Claude
# -----------------------------------------------------------------------------


class ClaudeConfig(BaseAgentConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Claude"
    model: str = Field(default="claude-sonnet-4-5", validation_alias=_model_alias)
    model_client: Any = None  # AsyncAnthropic | AsyncAnthropicBedrock
    max_tokens: int = 16384
    use_computer_beta: bool = True
    validate_api_key: bool = True


class ClaudeCreateParams(BaseCreateParams, ClaudeConfig):
    pass


# -----------------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------------


class GeminiConfig(BaseAgentConfig):
    """Configuration for GeminiAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Gemini"
    model: str = Field(default="gemini-3-pro-preview", validation_alias=_model_alias)
    model_client: Any = None  # genai.Client
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 8192
    validate_api_key: bool = True


class GeminiCreateParams(BaseCreateParams, GeminiConfig):
    pass


class GeminiCUAConfig(GeminiConfig):
    """Configuration for GeminiCUAAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "GeminiCUA"
    model: str = Field(
        default="gemini-2.5-computer-use-preview-10-2025", validation_alias=_model_alias
    )
    excluded_predefined_functions: list[str] = Field(default_factory=list)


class GeminiCUACreateParams(BaseCreateParams, GeminiCUAConfig):
    pass


# -----------------------------------------------------------------------------
# OpenAI
# -----------------------------------------------------------------------------


class OpenAIConfig(BaseAgentConfig):
    """Configuration for OpenAIAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "OpenAI"
    model: str = Field(default="gpt-5.1", validation_alias=_model_alias)
    model_client: Any = None  # AsyncOpenAI
    max_output_tokens: int | None = None
    temperature: float | None = None
    reasoning: Any = None  # openai Reasoning
    tool_choice: Any = None  # openai ToolChoice
    truncation: Literal["auto", "disabled"] | None = None
    parallel_tool_calls: bool | None = None
    validate_api_key: bool = True


class OpenAICreateParams(BaseCreateParams, OpenAIConfig):
    pass


class OpenAIChatConfig(BaseAgentConfig):
    """Configuration for OpenAIChatAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "OpenAI Chat"
    model: str = Field(default="gpt-5-mini", validation_alias=_model_alias)
    openai_client: Any = None  # AsyncOpenAI
    api_key: str | None = None
    base_url: str | None = None
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)


class OpenAIChatCreateParams(BaseCreateParams, OpenAIChatConfig):
    pass


# -----------------------------------------------------------------------------
# GLM CUA
# -----------------------------------------------------------------------------


class GLMCUAConfig(OpenAIChatConfig):
    """Configuration for GLMCUAAgent (GLM-4.5V GUI Agent)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "GLM CUA"
    model: str = Field(default="glm-4.5v", validation_alias=_model_alias)
    max_history_screenshots: int = 2  # Env: GLM_MAX_HISTORY_SCREENSHOTS
    history_image_scale: float = 0.25  # Env: GLM_HISTORY_IMAGE_SCALE (25%)


class GLMCUACreateParams(BaseCreateParams, GLMCUAConfig):
    pass


# -----------------------------------------------------------------------------
# Operator
# -----------------------------------------------------------------------------


class OperatorConfig(OpenAIConfig):
    """Configuration for OperatorAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Operator"
    model: str = Field(default="computer-use-preview", validation_alias=_model_alias)
    environment: Literal["windows", "mac", "linux", "ubuntu", "browser"] = "linux"


class OperatorCreateParams(BaseCreateParams, OperatorConfig):
    pass
