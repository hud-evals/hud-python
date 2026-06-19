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
        Customize settings source precedence.

        Precedence (highest to lowest):
        - init_settings (explicit kwargs)
        - env_settings (process environment)
        - dotenv_settings (.env in CWD)
        - user_dotenv_settings (~/.hud/.env, written by `hud set`)
        - file_secret_settings
        """
        user_dotenv_settings = DotEnvSettingsSource(
            settings_cls,
            env_file=Path.home() / ".hud" / ".env",
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
        default="https://telemetry.beta.hud.ai/v3/api",
        description="Base URL for the HUD API",
        validation_alias="HUD_TELEMETRY_URL",
    )

    hud_api_url: str = Field(
        default="https://api.beta.hud.ai",
        description="Base URL (origin) for the HUD API server",
        validation_alias="HUD_API_URL",
    )

    hud_web_url: str = Field(
        default="https://beta.hud.ai",
        description="Base URL of the HUD web app (used as a fallback for CLI login)",
        validation_alias="HUD_WEB_URL",
    )

    hud_gateway_url: str = Field(
        default="https://inference.beta.hud.ai",
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

    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS access key ID for Bedrock",
        validation_alias="AWS_ACCESS_KEY_ID",
    )

    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS secret access key for Bedrock",
        validation_alias="AWS_SECRET_ACCESS_KEY",
    )

    aws_region: str | None = Field(
        default=None,
        description="AWS region for Bedrock (e.g., us-east-1)",
        validation_alias="AWS_REGION",
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

    telemetry_local_dir: str | None = Field(
        default=None,
        description="If set, also write each telemetry span to <dir>/<trace_id>.jsonl "
        "locally. Independent of the backend exporter — works with no API key.",
        validation_alias="HUD_TELEMETRY_LOCAL_DIR",
    )

    file_tracking_enabled: bool = Field(
        default=False,
        description="Publish a workspace's filetracking/1 capability and stream file-change "
        "diffs to telemetry during a rollout. Opt-in; off by default.",
        validation_alias="HUD_FILE_TRACKING_ENABLED",
    )

    file_tracking_interval: float = Field(
        default=2.0,
        gt=0,
        description="Seconds between rollout-level file-tracking snapshots. Each snapshot "
        "diffs the workspace against the previous one and emits a hud.filetracking.v1 span.",
        validation_alias="HUD_FILE_TRACKING_INTERVAL",
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

    client_timeout: int = Field(
        default=600,
        ge=0,
        description=(
            "Global timeout in seconds for MCP requests "
            "(per-attempt timeout is configured separately; 0 uses default)"
        ),
        validation_alias="HUD_CLIENT_TIMEOUT",
    )


# Create a singleton instance
settings = Settings()


# Add utility functions for backwards compatibility
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
