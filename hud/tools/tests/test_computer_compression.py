"""Tests for image MIME detection on computer tool results."""

from __future__ import annotations

import base64
from io import BytesIO

from mcp.types import ImageContent
from PIL import Image

from hud.tools.types import ContentResult


def _make_png_base64(width: int = 10, height: int = 10) -> str:
    buf = BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class TestMimeTypeDetection:
    """ContentResult.to_content_blocks() labels image formats correctly."""

    def test_jpeg_image_gets_jpeg_mimetype(self):
        buf = BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="JPEG")
        jpeg_b64 = base64.b64encode(buf.getvalue()).decode()

        result = ContentResult(base64_image=jpeg_b64)
        blocks = result.to_content_blocks()

        img_block = next(b for b in blocks if isinstance(b, ImageContent))
        assert img_block.mimeType == "image/jpeg"

    def test_png_image_gets_png_mimetype(self):
        result = ContentResult(base64_image=_make_png_base64())
        blocks = result.to_content_blocks()

        img_block = next(b for b in blocks if isinstance(b, ImageContent))
        assert img_block.mimeType == "image/png"
