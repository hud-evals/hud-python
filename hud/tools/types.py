from __future__ import annotations

from typing import Any

from mcp.types import ContentBlock, ImageContent, TextContent
from pydantic import BaseModel, ConfigDict, Field


class Coordinate(BaseModel):
    """A coordinate point with x and y values.

    Used for path-based actions like drag operations.
    """

    model_config = ConfigDict(extra="forbid")

    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")


class SubScore(BaseModel):
    """Individual subscore for debugging and transparency.

    SubScores allow breaking down the final reward into component parts,
    making it easier to understand what contributed to the evaluation.

    Example:
        subscores=[
            SubScore(name="correctness", weight=0.6, value=1.0),
            SubScore(name="efficiency", weight=0.3, value=0.8),
            SubScore(name="style", weight=0.1, value=0.5),
        ]
        # Final reward could be: 0.6*1.0 + 0.3*0.8 + 0.1*0.5 = 0.89
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of this subscore component")
    weight: float = Field(default=1.0, description="Weight of this subscore (for weighted average)")
    value: float = Field(..., description="Value of this subscore, usually 0.0 to 1.0")


class EvaluationResult(BaseModel):
    """Standard evaluation result format.

    Used as the second yield in scenarios to provide detailed evaluation results.
    Can include subscores for debugging and transparency.

    Example:
        yield EvaluationResult(
            reward=0.85,
            done=True,
            content="Found 17 of 20 items",
            subscores=[
                SubScore(name="detection", weight=0.7, value=0.85),
                SubScore(name="accuracy", weight=0.3, value=1.0),
            ],
        )
    """

    reward: float = Field(default=0.0, description="Final score, usually 0.0 to 1.0")
    done: bool = Field(default=True, description="Whether the task/episode is complete")
    content: str | None = Field(default=None, description="Human-readable explanation")
    info: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    isError: bool = Field(default=False, description="Whether the evaluation itself failed")
    subscores: list[SubScore] | None = Field(
        default=None,
        description="Optional breakdown of score components for debugging",
    )

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_float(cls, value: float) -> EvaluationResult:
        """Create an EvaluationResult from a simple float reward.

        Convenience method for backward compatibility with float yields.
        Sets done=True since a float yield typically indicates completion.
        """
        return cls(reward=value, done=True)


class ContentResult(BaseModel):
    """Represents the intermediate result of a tool execution.

    Often useful for tools that need to return multiple types of content.
    """

    output: str | None = Field(default=None, description="Output text")
    error: str | None = Field(default=None, description="Error message")
    base64_image: str | None = Field(default=None, description="Base64-encoded image")
    system: str | None = Field(default=None, description="System message")
    url: str | None = Field(default=None, description="Current page URL (for browser automation)")

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
            url=combine_fields(self.url, other.url, False),
        )

    def to_text_blocks(self) -> list[TextContent]:
        """Convert text-only content to TextContent blocks.

        Use this for tools that only return text output.

        Returns:
            List of TextContent blocks
        """
        blocks: list[TextContent] = []

        if self.output:
            blocks.append(TextContent(text=self.output, type="text"))
        if self.error:
            blocks.append(TextContent(text=self.error, type="text"))
        if self.url:
            blocks.append(TextContent(text=f"__URL__:{self.url}", type="text"))

        return blocks

    def to_content_blocks(self) -> list[ContentBlock]:
        """Convert to content blocks including images.

        Use to_text_blocks() for text-only tools for better type safety.

        Returns:
            List of ContentBlock with URL embedded as metadata if available
        """
        blocks: list[ContentBlock] = list(self.to_text_blocks())

        if self.base64_image:
            blocks.append(ImageContent(data=self.base64_image, mimeType="image/png", type="image"))

        return blocks


class ToolError(Exception):
    """An error raised by a tool."""
