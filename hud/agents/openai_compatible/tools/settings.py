"""OpenAI-compatible native tool settings owned by the agent."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAICompatibleToolSettings(BaseSettings):
    """Provider defaults for OpenAI-compatible agent-owned native tools."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    GLM_COMPUTER_WIDTH: int = Field(
        default=1024,
        description="Default GLM computer-use display width",
        validation_alias="GLM_COMPUTER_WIDTH",
    )
    GLM_COMPUTER_HEIGHT: int = Field(
        default=768,
        description="Default GLM computer-use display height",
        validation_alias="GLM_COMPUTER_HEIGHT",
    )
    QWEN_COMPUTER_WIDTH: int = Field(
        default=700,
        description="Default Qwen computer-use display width",
        validation_alias="QWEN_COMPUTER_WIDTH",
    )
    QWEN_COMPUTER_HEIGHT: int = Field(
        default=448,
        description="Default Qwen computer-use display height",
        validation_alias="QWEN_COMPUTER_HEIGHT",
    )


openai_compatible_tool_settings = OpenAICompatibleToolSettings()
