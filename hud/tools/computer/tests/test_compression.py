"""Tests for JPEG screenshot compression in AnthropicComputerTool."""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import AsyncMock

import pytest
from mcp.types import ImageContent
from PIL import Image

from hud.tools.computer.anthropic import AnthropicComputerTool
from hud.tools.types import ContentResult


def _make_png_base64(width: int = 200, height: int = 150, mode: str = "RGB") -> str:
    """Create a realistic PNG image with pixel noise (not solid color).

    Solid-color PNGs compress to almost nothing, making them smaller than
    any JPEG.  Real screenshots have gradients and noise, so we simulate
    that here to get representative PNG-vs-JPEG size behaviour.
    """
    import random

    random.seed(42)
    channels = 4 if mode == "RGBA" else 3
    pixels = bytes(random.randint(0, 255) for _ in range(width * height * channels))
    img = Image.frombytes(mode, (width, height), pixels)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_image(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64)))


class TestScreenshotCompression:
    """Core compression: PNG in, smaller JPEG out."""

    @pytest.mark.asyncio
    async def test_compression_produces_smaller_jpeg(self):
        tool = AnthropicComputerTool(screenshot_quality=60)
        png_b64 = _make_png_base64(1024, 768)

        result = await tool._rescale_screenshot(png_b64)

        assert len(result) < len(png_b64), "JPEG should be smaller than PNG"
        img = _decode_image(result)
        assert img.format == "JPEG"

    @pytest.mark.asyncio
    async def test_no_compression_when_quality_is_none(self):
        tool = AnthropicComputerTool(screenshot_quality=None)
        png_b64 = _make_png_base64(200, 150)

        result = await tool._rescale_screenshot(png_b64)

        img = _decode_image(result)
        assert img.format == "PNG"


class TestRGBAConversion:
    """JPEG doesn't support transparency — RGBA PNGs must be converted."""

    @pytest.mark.asyncio
    async def test_rgba_png_compresses_without_error(self):
        tool = AnthropicComputerTool(screenshot_quality=60)
        rgba_b64 = _make_png_base64(400, 300, mode="RGBA")

        result = await tool._rescale_screenshot(rgba_b64)

        img = _decode_image(result)
        assert img.format == "JPEG"
        assert img.mode == "RGB"


class TestZoomCompression:
    """Zoom crops should be compressed but never resized."""

    @pytest.mark.asyncio
    async def test_zoom_preserves_dimensions_but_compresses(self):
        crop_w, crop_h = 300, 250
        tool = AnthropicComputerTool(
            screenshot_quality=60,
            width=1400,
            height=850,
        )
        png_b64 = _make_png_base64(crop_w, crop_h)

        result = await tool._rescale_screenshot(png_b64, skip_resize=True)

        img = _decode_image(result)
        assert img.format == "JPEG"
        assert img.size == (crop_w, crop_h), "Zoom crop dimensions must not change"

    @pytest.mark.asyncio
    async def test_zoom_action_routes_through_skip_resize(self):
        """End-to-end: a zoom action compresses without resizing."""
        crop_w, crop_h = 300, 250
        tool = AnthropicComputerTool(screenshot_quality=60)

        zoom_result = ContentResult(base64_image=_make_png_base64(crop_w, crop_h))
        tool.executor.zoom = AsyncMock(return_value=zoom_result)

        blocks = await tool(action="zoom", region=[0, 0, 400, 400])

        img_block = next(b for b in blocks if isinstance(b, ImageContent))
        img = _decode_image(img_block.data)
        assert img.format == "JPEG"
        assert img.size == (crop_w, crop_h)


class TestResizeAndCompress:
    """When screen > agent coords, screenshots get both resized and compressed."""

    @pytest.mark.asyncio
    async def test_large_screenshot_gets_resized_and_compressed(self):
        tool = AnthropicComputerTool(
            screenshot_quality=60,
            width=1400,
            height=850,
            rescale_images=True,
        )
        # Simulate a screen larger than agent coordinates
        tool.environment_width = 1920
        tool.environment_height = 1080
        tool.scale_x = tool.width / tool.environment_width
        tool.scale_y = tool.height / tool.environment_height
        tool.needs_scaling = True

        big_png = _make_png_base64(1920, 1080)

        result = await tool._rescale_screenshot(big_png)

        img = _decode_image(result)
        assert img.format == "JPEG"
        assert img.size == (1400, 850), "Should be resized to agent dimensions"
        assert len(result) < len(big_png)


class TestMimeTypeDetection:
    """ContentResult.to_content_blocks() must label JPEG vs PNG correctly."""

    def test_jpeg_image_gets_jpeg_mimetype(self):
        buf = BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="JPEG")
        jpeg_b64 = base64.b64encode(buf.getvalue()).decode()

        result = ContentResult(base64_image=jpeg_b64)
        blocks = result.to_content_blocks()

        img_block = next(b for b in blocks if isinstance(b, ImageContent))
        assert img_block.mimeType == "image/jpeg"

    def test_png_image_gets_png_mimetype(self):
        png_b64 = _make_png_base64(10, 10)

        result = ContentResult(base64_image=png_b64)
        blocks = result.to_content_blocks()

        img_block = next(b for b in blocks if isinstance(b, ImageContent))
        assert img_block.mimeType == "image/png"
