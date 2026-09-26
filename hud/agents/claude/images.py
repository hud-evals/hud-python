"""Image input preparation for Anthropic Messages."""

from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import TYPE_CHECKING, Literal, cast

from anthropic.types.beta import BetaBase64ImageSourceParam
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from mcp.types import ImageContent

ClaudeImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
MAX_SOURCE_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_EDGE = 1568
MAX_IMAGE_PIXELS = 1_150_000
MAX_IMAGE_BASE64_BYTES = 5_000_000


def image_source(
    content: ImageContent, *, preserve_dimensions: bool = False
) -> BetaBase64ImageSourceParam:
    """Prepare a tool image, preserving computer-tool coordinate space when requested."""
    if preserve_dimensions:
        return BetaBase64ImageSourceParam(
            type="base64",
            media_type=cast("ClaudeImageMediaType", content.mimeType),
            data=content.data,
        )
    with Image.open(BytesIO(base64.b64decode(content.data, validate=True))) as image:
        if image.width * image.height > MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError(
                f"Image exceeds the {MAX_SOURCE_IMAGE_PIXELS:,}-pixel source limit. "
                "Resize or crop it before viewing."
            )
        return _encode_image(image)


def _encode_image(image: Image.Image) -> BetaBase64ImageSourceParam:
    scale = min(
        1.0,
        MAX_IMAGE_EDGE / max(image.size),
        math.sqrt(MAX_IMAGE_PIXELS / (image.width * image.height)),
    )
    image.thumbnail(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGBA" if image.has_transparency_data else "RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type: ClaudeImageMediaType = "image/png"
    if len(data) > MAX_IMAGE_BASE64_BYTES:
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A") if image.mode == "RGBA" else None)
        buffer = BytesIO()
        background.save(buffer, format="JPEG", quality=85, optimize=True)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        mime_type = "image/jpeg"
    return BetaBase64ImageSourceParam(type="base64", media_type=mime_type, data=data)
