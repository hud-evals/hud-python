from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ComputerSettings(BaseSettings):
    """
    Local computer settings for the HUD SDK.

    This class manages local computer settings for the HUD SDK.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    DISPLAY_WIDTH: int = Field(
        default=1920,
        description="Width of the display to use for the computer tools",
        validation_alias="DISPLAY_WIDTH",
    )
    DISPLAY_HEIGHT: int = Field(
        default=1080,
        description="Height of the display to use for the computer tools",
        validation_alias="DISPLAY_HEIGHT",
    )
    DISPLAY_NUM: int = Field(
        default=0,
        description="Number of the display to use for the computer tools",
        validation_alias="DISPLAY_NUM",
    )

    HUD_COMPUTER_WIDTH: int = Field(
        default=1920,
        description="Width of the display to use for the computer tools",
        validation_alias="HUD_COMPUTER_WIDTH",
    )
    HUD_COMPUTER_HEIGHT: int = Field(
        default=1080,
        description="Height of the display to use for the computer tools",
        validation_alias="HUD_COMPUTER_HEIGHT",
    )

    ANTHROPIC_COMPUTER_WIDTH: int = Field(
        default=1400,
        description="Width of the display to use for the Anthropic computer tools",
        validation_alias="ANTHROPIC_COMPUTER_WIDTH",
    )
    ANTHROPIC_COMPUTER_HEIGHT: int = Field(
        default=850,
        description="Height of the display to use for the Anthropic computer tools",
        validation_alias="ANTHROPIC_COMPUTER_HEIGHT",
    )

    OPENAI_COMPUTER_WIDTH: int = Field(
        default=1024,
        description="Width of the display to use for the OpenAI computer tools",
        validation_alias="OPENAI_COMPUTER_WIDTH",
    )
    OPENAI_COMPUTER_HEIGHT: int = Field(
        default=768,
        description="Height of the display to use for the OpenAI computer tools",
        validation_alias="OPENAI_COMPUTER_HEIGHT",
    )

    HUD_RESCALE_IMAGES: bool = Field(
        default=False,
        description="Whether to rescale images to the agent width and height",
        validation_alias="HUD_RESCALE_IMAGES",
    )
    ANTHROPIC_RESCALE_IMAGES: bool = Field(
        default=True,
        description="Whether to rescale images to the agent width and height",
        validation_alias="ANTHROPIC_RESCALE_IMAGES",
    )
    OPENAI_RESCALE_IMAGES: bool = Field(
        default=False,
        description="Whether to rescale images to the agent width and height",
        validation_alias="OPENAI_RESCALE_IMAGES",
    )


computer_settings = ComputerSettings()
