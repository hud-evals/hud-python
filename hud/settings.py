from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global settings for the HUD SDK.

    This class manages configuration values loaded from environment variables
    and provides global access to settings throughout the application.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    hud_telemetry_url: str = Field(
        default="https://telemetry.hud.so/v3/api",
        description="Base URL for the HUD API",
        validation_alias="HUD_TELEMETRY_URL",
    )

    hud_mcp_url: str = Field(
        default="https://mcp.hud.so/v3/mcp",
        description="Base URL for the MCP Server",
        validation_alias="HUD_MCP_URL",
    )

    api_key: str | None = Field(
        default=None,
        description="API key for authentication with the HUD API",
        validation_alias="HUD_API_KEY",
    )

    anthropic_api_key: str | None = Field(
        default=None,
        description="API key for Anthropic models",
        validation_alias="ANTHROPIC_API_KEY",
    )

    openai_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI models",
        validation_alias="OPENAI_API_KEY",
    )

    openrouter_api_key: str | None = Field(
        default=None,
        description="API key for OpenRouter models",
        validation_alias="OPENROUTER_API_KEY",
    )

    wandb_api_key: str | None = Field(
        default=None,
        description="API key for Weights & Biases",
        validation_alias="WANDB_API_KEY",
    )

    prime_api_key: str | None = Field(
        default=None,
        description="API key for Prime Intellect",
        validation_alias="PRIME_API_KEY",
    )

    telemetry_enabled: bool = Field(
        default=True,
        description="Enable telemetry for the HUD SDK",
        validation_alias="HUD_TELEMETRY_ENABLED",
    )

    hud_logging: bool = Field(
        default=True,
        description="Enable fancy logging for the HUD SDK",
        validation_alias="HUD_LOGGING",
    )

    log_stream: str = Field(
        default="stdout",
        description="Stream to use for logging output: 'stdout' or 'stderr'",
        validation_alias="HUD_LOG_STREAM",
    )


# Create a singleton instance
settings = Settings()


# Add utility functions for backwards compatibility
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
