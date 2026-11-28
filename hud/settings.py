from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource, PydanticBaseSettingsSource


class Settings(BaseSettings):
    """Global settings for the HUD SDK."""

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
        """Layer ``~/.hud/.env`` (from ``hud set``) under the process env and the CWD ``.env``."""
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
        default="https://telemetry.hud.ai/v3/api",
        description="Base URL for HUD telemetry ingestion",
        validation_alias="HUD_TELEMETRY_URL",
    )

    hud_api_url: str = Field(
        default="https://api.hud.ai",
        description="Base URL (origin) for the HUD API server",
        validation_alias="HUD_API_URL",
    )

    hud_web_url: str = Field(
        default="https://hud.ai",
        description="Base URL of the HUD web app, used for the links the CLI and job logs print",
        validation_alias="HUD_WEB_URL",
    )

    hud_gateway_url: str = Field(
        default="https://inference.hud.ai",
        description="Base URL for the HUD inference gateway",
        validation_alias="HUD_GATEWAY_URL",
    )

    hud_runtime_url: str = Field(
        default="https://mcp.hud.ai",
        description="Base URL for the HUD runtime tunnel gateway",
        validation_alias="HUD_RUNTIME_URL",
    )

    api_key: str | None = Field(
        default=None,
        description="API key for authentication with the HUD API",
        validation_alias="HUD_API_KEY",
    )

    default_project: str | None = Field(
        default=None,
        description="Default HUD Project name or ID for directories without a saved Project",
        validation_alias="HUD_DEFAULT_PROJECT",
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

    telemetry_enabled: bool = Field(
        default=True,
        description="Upload runs and their traces to the HUD platform",
        validation_alias="HUD_TELEMETRY_ENABLED",
    )

    cli_analytics_enabled: bool = Field(
        default=True,
        description="Send anonymous HUD CLI usage analytics",
        validation_alias="HUD_CLI_ANALYTICS_ENABLED",
    )

    trace_dir: Path | None = Field(
        default_factory=lambda: Path.home() / ".hud" / "traces",
        description="Directory for local trace files; empty writes none.",
        validation_alias=AliasChoices("HUD_TRACE_DIR", "HUD_TELEMETRY_LOCAL_DIR"),
    )

    file_tracking_enabled: bool = Field(
        default=True,
        description="Publish a workspace's filetracking/1 capability and stream setup changes, "
        "agent file-change diffs, and final artifacts to telemetry.",
        validation_alias="HUD_FILE_TRACKING_ENABLED",
    )

    file_tracking_interval: float = Field(
        default=2.0,
        gt=0,
        description="Seconds between rollout-level file-tracking snapshots. Each snapshot "
        "diffs the workspace against the previous one and emits a hud.filetracking.v1 span.",
        validation_alias="HUD_FILE_TRACKING_INTERVAL",
    )

    @field_validator("trace_dir", mode="before")
    @classmethod
    def _empty_trace_dir_is_off(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def reporting_enabled(self) -> bool:
        """Whether runs and spans reach the platform: telemetry is on and an API key is set."""
        return bool(self.telemetry_enabled and self.api_key)


settings = Settings()
