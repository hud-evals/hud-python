"""Claude native tool settings owned by the Claude agent."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClaudeToolSettings(BaseSettings):
    """Claude provider defaults for agent-owned native tools."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    COMPUTER_WIDTH: int = Field(
        default=1400,
        description="Default Claude computer-use display width",
        validation_alias="ANTHROPIC_COMPUTER_WIDTH",
    )
    COMPUTER_HEIGHT: int = Field(
        default=850,
        description="Default Claude computer-use display height",
        validation_alias="ANTHROPIC_COMPUTER_HEIGHT",
    )
    RESCALE_IMAGES: bool = Field(
        default=True,
        description="Whether Claude computer screenshots should be rescaled",
        validation_alias="ANTHROPIC_RESCALE_IMAGES",
    )
    SCREENSHOT_QUALITY: int | None = Field(
        default=None,
        description="JPEG quality for Claude screenshots. None keeps lossless PNG.",
        validation_alias="ANTHROPIC_SCREENSHOT_QUALITY",
    )


claude_tool_settings = ClaudeToolSettings()

__all__ = ["ClaudeToolSettings", "claude_tool_settings"]
