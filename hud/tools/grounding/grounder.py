"""OpenAI-based grounder for visual element detection."""

from __future__ import annotations

import base64
import io
import json
import re

from openai import AsyncOpenAI
from opentelemetry import trace

from hud import instrument
from hud.tools.grounding.config import GrounderConfig  # noqa: TC001


class Grounder:
    """Grounder that uses AsyncOpenAI to call vLLM or other model endpoints for visual grounding.

    This class handles:
    - Image resizing based on configuration
    - API calls to grounding models via AsyncOpenAI
    - Coordinate parsing from model outputs
    - Coordinate format conversion (pixels, normalized)
    """

    def __init__(self, config: GrounderConfig) -> None:
        """Initialize the grounder with configuration.

        Args:
            config: GrounderConfig with API endpoint, model, and parsing settings
        """
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.api_base)

    def _resize_image(self, image_b64: str) -> tuple[str, tuple[int, int], tuple[int, int]]:
        """Resize image according to configuration.

        Args:
            image_b64: Base64-encoded image string

        Returns:
            Tuple of (processed_base64, (original_width, original_height),
                     (processed_width, processed_height))
        """
        # Decode image
        from PIL import Image

        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes))
        original_size = (img.width, img.height)

        if not self.config.resize["enabled"]:
            return image_b64, original_size, original_size

        # Calculate total pixels
        total_pixels = img.width * img.height
        min_pixels = self.config.resize["min_pixels"]
        max_pixels = self.config.resize["max_pixels"]
        factor = self.config.resize["factor"]

        # Determine if resizing is needed
        if total_pixels < min_pixels or total_pixels > max_pixels:
            # Calculate scaling factor
            if total_pixels < min_pixels:
                scale = (min_pixels / total_pixels) ** 0.5
            else:
                scale = (max_pixels / total_pixels) ** 0.5

            # Round dimensions to nearest factor
            new_width = int((img.width * scale) // factor) * factor
            new_height = int((img.height * scale) // factor) * factor

            # Ensure minimum dimensions
            new_width = max(new_width, factor)
            new_height = max(new_height, factor)

            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert back to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            resized_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return resized_b64, original_size, (new_width, new_height)

        return image_b64, original_size, original_size

    def _parse_coordinates(self, response_text: str) -> tuple[float, float] | None:
        """Parse coordinates from model response.

        Handles multiple formats:
        - (x, y) format from configured regex
        - [x1, y1, x2, y2] bounding box format (returns center point)
        - [x, y] point format

        Args:
            response_text: Text output from the grounding model

        Returns:
            Tuple of (x, y) coordinates or None if parsing fails
        """
        # First try the configured regex pattern
        match = re.search(self.config.parser_regex, response_text)
        if match:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                return (x, y)
            except (ValueError, IndexError):
                # If parsing fails, continue to fallback strategies
                pass

        # Try to parse as a list/array format [x1, y1, x2, y2] or [x, y]
        # Also handles (x1, y1, x2, y2)
        # Updated pattern to handle both integers and floats
        list_pattern = (
            r"[\[\(](\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)"
            r"(?:[,\s]+(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?))?[\]\)]"
        )
        list_match = re.search(list_pattern, response_text)
        if list_match:
            x1 = float(list_match.group(1))
            y1 = float(list_match.group(2))

            # Check if it's a bounding box (4 values) or a point (2 values)
            if list_match.group(3) and list_match.group(4):
                # Bounding box format - return center point
                x2 = float(list_match.group(3))
                y2 = float(list_match.group(4))
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                return (center_x, center_y)
            else:
                # Point format
                return (x1, y1)

        return None

    def _convert_coordinates(
        self,
        coords: tuple[float, float],
        processed_size: tuple[int, int],
        original_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Convert coordinates based on output format configuration and scale to original size.

        Args:
            coords: Raw coordinates from model (can be float for normalized formats)
            processed_size: Dimensions of the processed/resized image (width, height)
            original_size: Original image dimensions (width, height)

        Returns:
            Converted coordinates in original image pixels
        """
        x, y = coords
        proc_width, proc_height = processed_size
        orig_width, orig_height = original_size

        # First convert to pixels in the processed image space
        if self.config.output_format == "pixels":
            # Already in pixels of processed image
            proc_x, proc_y = x, y
        elif self.config.output_format == "norm_0_1":
            # Convert from 0-1 normalized to pixels
            proc_x = x * proc_width
            proc_y = y * proc_height
        elif self.config.output_format == "norm_0_999":
            # Convert from 0-999 normalized to pixels
            proc_x = x * proc_width / 999
            proc_y = y * proc_height / 999
        else:
            proc_x, proc_y = x, y

        # Scale from processed image coordinates to original image coordinates
        scale_x = orig_width / proc_width
        scale_y = orig_height / proc_height

        final_x = int(proc_x * scale_x)
        final_y = int(proc_y * scale_y)

        return (final_x, final_y)

    @instrument(
        name="Grounding.predict_click",
        span_type="agent",
        record_args=True,
        record_result=True,
    )
    async def predict_click(
        self, *, image_b64: str, instruction: str, max_retries: int = 3
    ) -> tuple[int, int] | None:
        """Predict click coordinates for the given instruction on the image.

        Args:
            image_b64: Base64-encoded screenshot
            instruction: Natural language description of the element to click
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            Tuple of (x, y) pixel coordinates or None if grounding fails
        """

        # Resize image once outside the retry loop
        processed_image, original_size, processed_size = self._resize_image(image_b64)

        # Build messages once
        messages = []

        # Add system prompt if configured
        if self.config.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        self.config.system_prompt
                        + f" The image resolution is height {processed_size[1]} "
                        + f"and width {processed_size[0]}."
                    ),
                }
            )

        # Add user message with image and instruction
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{processed_image}"},
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        )

        # Retry loop
        for attempt in range(max_retries):
            try:
                # Call the grounding model via AsyncOpenAI
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=50,
                )

                # Extract response text
                response_text = response.choices[0].message.content

                # Manually record the raw response in the span
                span = trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("grounder.raw_response", json.dumps(response.model_dump()))
                    span.set_attribute("grounder.attempt", attempt + 1)

                # Parse coordinates from response
                if response_text is None:
                    if attempt < max_retries - 1:
                        continue
                    return None

                coords = self._parse_coordinates(response_text)
                if coords is None:
                    if attempt < max_retries - 1:
                        continue
                    return None

                # Convert coordinates to original image pixels based on output format and scaling
                pixel_coords = self._convert_coordinates(coords, processed_size, original_size)

                # Validate coordinates are within image bounds
                x, y = pixel_coords
                if x < 0 or y < 0 or x >= original_size[0] or y >= original_size[1]:
                    # Clamp to image bounds
                    x = max(0, min(x, original_size[0] - 1))
                    y = max(0, min(y, original_size[1] - 1))
                    pixel_coords = (x, y)

                # Record successful grounding in span
                span = trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("grounder.success", True)
                    span.set_attribute(
                        "grounder.final_coords", f"{pixel_coords[0]},{pixel_coords[1]}"
                    )
                    span.set_attribute("grounder.total_attempts", attempt + 1)

                return pixel_coords

            except Exception:
                if attempt < max_retries - 1:
                    continue

        # Record failure in span
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("grounder.success", False)
            span.set_attribute("grounder.total_attempts", max_retries)
            span.set_attribute("grounder.failure_reason", "All attempts exhausted")

        return None
