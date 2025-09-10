"""Configuration for grounding models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SYSTEM_PROMPT = (
    "You are a visual grounding model. Given an image and a description, "
    "return ONLY the center pixel coordinates of the described element as a "
    "single point in parentheses format: (x, y). Do not return bounding boxes "
    "or multiple coordinates."
)


@dataclass
class GrounderConfig:
    """Configuration for grounding model clients.

    Attributes:
        api_base: Base URL for the grounding model API endpoint
        model: Model identifier to use for grounding
        api_key: API key for authentication (default: "EMPTY" for local models)
        system_prompt: System prompt to guide the grounding model
        output_format: Format for coordinate output ("pixels", "norm_0_1", "norm_0_999")
        parser_regex: Regular expression to parse coordinates from model output
        resize: Image resizing configuration dictionary
    """

    api_base: str
    model: str
    api_key: str = "EMPTY"
    system_prompt: str = SYSTEM_PROMPT
    output_format: str = "pixels"  # "pixels" | "norm_0_1" | "norm_0_999"
    parser_regex: str = r"\((\d+),\s*(\d+)\)"
    resize: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "min_pixels": 3136,
            "max_pixels": 4096 * 2160,
            "factor": 28,
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.output_format not in ("pixels", "norm_0_1", "norm_0_999"):
            raise ValueError(f"Invalid output_format: {self.output_format}")

        if not self.api_base:
            raise ValueError("api_base is required")

        if not self.model:
            raise ValueError("model is required")
