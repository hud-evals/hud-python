"""Text and image content for agent file views."""

from __future__ import annotations

import base64
import math
from io import BytesIO

import mcp.types as mcp_types
from PIL import Image, ImageOps, UnidentifiedImageError

from hud.agents.tools.base import tool_err, tool_ok
from hud.types import MCPToolResult

MAX_SOURCE_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_EDGE = 1568
MAX_IMAGE_PIXELS = 1_150_000
MAX_IMAGE_BASE64_BYTES = 5_000_000


def view_file(data: bytes) -> MCPToolResult:
    """Decode file bytes into text or a bounded image tool result."""
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"PNG", "JPEG", "GIF", "WEBP"}:
                return tool_err(
                    f"Unsupported image type: {image.format}. Convert it to PNG or JPEG."
                )
            if image.width * image.height > MAX_SOURCE_IMAGE_PIXELS:
                return tool_err(
                    f"Image exceeds the {MAX_SOURCE_IMAGE_PIXELS:,}-pixel source limit. "
                    "Resize or crop it before viewing."
                )
            content = _image_content(image)
    except UnidentifiedImageError:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return tool_err("Cannot view this file: expected UTF-8 text or an image.")
        if "\x00" in text:
            return tool_err("Cannot view this file: expected UTF-8 text or an image.")
        return tool_ok(text)
    except (OSError, Image.DecompressionBombError) as e:
        return tool_err(f"Cannot view this image: {e}")
    return MCPToolResult(content=[content])


def _image_content(image: Image.Image) -> mcp_types.ImageContent:
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
    mime_type = "image/png"
    if len(data) > MAX_IMAGE_BASE64_BYTES:
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A") if image.mode == "RGBA" else None)
        buffer = BytesIO()
        background.save(buffer, format="JPEG", quality=85, optimize=True)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        mime_type = "image/jpeg"
    return mcp_types.ImageContent(type="image", mimeType=mime_type, data=data)
