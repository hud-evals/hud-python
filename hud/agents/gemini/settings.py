"""Gemini agent settings."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiAgentSettings(BaseSettings):
    """Gemini provider defaults owned by the agent."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    MAX_RECENT_TURN_WITH_SCREENSHOTS: int = Field(
        default=3,
        description="Maximum number of recent turns to keep screenshots for in Gemini agent",
        validation_alias="GEMINI_MAX_RECENT_TURN_WITH_SCREENSHOTS",
    )


gemini_agent_settings = GeminiAgentSettings()
