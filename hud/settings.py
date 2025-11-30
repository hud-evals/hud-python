from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource, PydanticBaseSettingsSource


class Settings(BaseSettings):
    """
    Global settings for the HUD SDK.

    This class manages configuration values loaded from environment variables
    and provides global access to settings throughout the application.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize settings source precedence to include a user-level env file.

        Precedence (highest to lowest):
        - init_settings (explicit kwargs)
        - env_settings (process environment)
        - dotenv_settings (project .env)
        - user_dotenv_settings (~/.hud/.env)  ← added
        - file_secret_settings
        """

        user_env_path = Path.home() / ".hud" / ".env"
        user_dotenv_settings = DotEnvSettingsSource(
            settings_cls,
            env_file=user_env_path,
            env_file_encoding="utf-8",
        )

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            user_dotenv_settings,
            file_secret_settings,
        )

    hud_telemetry_url: str = Field(
        default="https://telemetry.hud.ai/v3/api",
        description="Base URL for the HUD API",
        validation_alias="HUD_TELEMETRY_URL",
    )

    hud_mcp_url: str = Field(
        default="https://mcp.hud.ai/v3/mcp",
        description="Base URL for the MCP Server",
        validation_alias="HUD_MCP_URL",
    )

    hud_rl_url: str = Field(
        default="https://rl.hud.ai/v1",
        description="Base URL for the HUD RL API server",
        validation_alias="HUD_RL_URL",
    )

    hud_api_url: str = Field(
        default="https://api.hud.ai",
        description="Base URL for the HUD API server",
        validation_alias="HUD_API_URL",
    )

    hud_gateway_url: str = Field(
        default="https://inference.hud.ai",
        description="Base URL for the HUD inference gateway",
        validation_alias="HUD_GATEWAY_URL",
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

    gemini_api_key: str | None = Field(
        default=None,
        description="API key for Google Gemini models",
        validation_alias="GEMINI_API_KEY",
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
