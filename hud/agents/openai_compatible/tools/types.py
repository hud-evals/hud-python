"""Type definitions for OpenAI-compatible chat tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionToolParam
    from openai.types.shared_params.function_parameters import FunctionParameters


class QwenComputerUseToolParam(TypedDict):
    """Qwen's OpenAI-compatible computer_use extension."""

    type: Literal["computer_use"]
    name: str
    display_width_px: int
    display_height_px: int
    description: str
    parameters: FunctionParameters


OpenAICompatibleToolParam: TypeAlias = "ChatCompletionToolParam | QwenComputerUseToolParam"


__all__ = ["OpenAICompatibleToolParam", "QwenComputerUseToolParam"]
