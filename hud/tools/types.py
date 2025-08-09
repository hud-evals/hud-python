from __future__ import annotations

from typing import Any

from mcp.types import ContentBlock, ImageContent, TextContent
from pydantic import BaseModel, ConfigDict, Field


class SetupResult(BaseModel):
    """Standard setup result format."""

    content: str | None = Field(default=None, description="Message")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional information")
    isError: bool = Field(default=False, description="Whether the setup failed")

    model_config = ConfigDict(extra="allow")


class EvaluationResult(BaseModel):
    """Standard evaluation result format."""

    reward: float = Field(default=0.0, description="Usually a value between 0.0 and 1.0")
    done: bool = Field(default=False, description="Whether the task/episode is complete")
    content: str | None = Field(default=None, description="Additional information")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional information")
    isError: bool = Field(default=False, description="Whether the evaluation failed")

    model_config = ConfigDict(extra="allow")


class ContentResult(BaseModel):
    """Represents the intermediate result of a tool execution.

    Often useful for tools that need to return multiple types of content.
    """

    output: str | None = Field(default=None, description="Output text")
    error: str | None = Field(default=None, description="Error message")
    base64_image: str | None = Field(default=None, description="Base64-encoded image")
    system: str | None = Field(default=None, description="System message")

    def __add__(self, other: ContentResult) -> ContentResult:
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ) -> str | None:
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ContentResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def to_content_blocks(self) -> list[ContentBlock]:
        """Helper method to convert ContentResult to content blocks.

        Subclasses can use this when they work with ContentResult internally.

        Args:
            result: ContentResult to convert

        Returns:
            List of ContentBlock
        """
        blocks: list[ContentBlock] = []

        if self.output:
            blocks.append(TextContent(text=self.output, type="text"))
        if self.error:
            blocks.append(TextContent(text=self.error, type="text"))
        if self.base64_image:
            blocks.append(ImageContent(data=self.base64_image, mimeType="image/png", type="image"))
        return blocks


class ToolError(Exception):
    """An error raised by a tool."""
