"""Screenshot geometry for Claude's computer toolset."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO

from mcp.types import ImageContent
from PIL import Image

from hud.agents.claude.images import MAX_IMAGE_BASE64_BYTES
from hud.capabilities.rfb import ScreenshotEncoding, WebPScreenshotEncoding

# Requests with more than 20 images cap each side at 2000 px. Apply that limit
# from the first turn so signed screenshot history never needs resizing.
_MAX_IMAGE_EDGE = 2000
_MAX_VISUAL_TOKENS = 4784
_PATCH_SIZE = 28


@dataclass(frozen=True)
class ComputerGeometry:
    screen_width: int
    screen_height: int
    screenshot_size: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "screenshot_size",
            _fit_size(self.screen_width, self.screen_height, (_MAX_IMAGE_EDGE, _MAX_IMAGE_EDGE)),
        )

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Map a screenshot pixel to the native display."""
        width, height = self.screenshot_size
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"coordinate {(x, y)} is outside the {width}x{height} screenshot")
        return x * self.screen_width // width, y * self.screen_height // height

    def from_screen(self, x: int, y: int) -> tuple[int, int]:
        """Map the native cursor position to the full screenshot's pixel space."""
        if not (0 <= x < self.screen_width and 0 <= y < self.screen_height):
            raise ValueError(f"cursor {(x, y)} is outside the native display")
        width, height = self.screenshot_size
        return x * width // self.screen_width, y * height // self.screen_height

    def screen_region(self, region: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Map an exclusive screenshot rectangle to a covering native rectangle."""
        x0, y0, x1, y1 = region
        width, height = self.screenshot_size
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"region {region} is outside the {width}x{height} screenshot")
        return (
            x0 * self.screen_width // width,
            y0 * self.screen_height // height,
            -(-x1 * self.screen_width // width),
            -(-y1 * self.screen_height // height),
        )

    def screenshot(self, data: bytes, encoding: ScreenshotEncoding) -> ImageContent:
        return _encode(self._load_screenshot(data), self.screenshot_size, encoding)

    def zoom(
        self,
        data: bytes,
        region: tuple[int, int, int, int],
        encoding: ScreenshotEncoding,
    ) -> ImageContent:
        image = self._load_screenshot(data).crop(self.screen_region(region))
        size = _fit_size(image.width, image.height, self.screenshot_size)
        return _encode(image, size, encoding)

    def _load_screenshot(self, data: bytes) -> Image.Image:
        with Image.open(BytesIO(data)) as image:
            if image.size != (self.screen_width, self.screen_height):
                raise ValueError(
                    f"screenshot size {image.size} does not match the native display "
                    f"{(self.screen_width, self.screen_height)}"
                )
            return image.convert("RGBA" if image.has_transparency_data else "RGB")


def _fit_size(width: int, height: int, bounds: tuple[int, int]) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("display dimensions must be positive")
    long_edge = max(width, height)
    low, high = 1, min(long_edge, bounds[0] * long_edge // width, bounds[1] * long_edge // height)
    while low < high:
        candidate = (low + high + 1) // 2
        scaled_width = max(1, width * candidate // long_edge)
        scaled_height = max(1, height * candidate // long_edge)
        tokens = ((scaled_width + _PATCH_SIZE - 1) // _PATCH_SIZE) * (
            (scaled_height + _PATCH_SIZE - 1) // _PATCH_SIZE
        )
        if tokens <= _MAX_VISUAL_TOKENS:
            low = candidate
        else:
            high = candidate - 1
    return max(1, width * low // long_edge), max(1, height * low // long_edge)


def _encode(
    image: Image.Image, size: tuple[int, int], encoding: ScreenshotEncoding
) -> ImageContent:
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    if isinstance(encoding, WebPScreenshotEncoding):
        image.save(buffer, format="WEBP", quality=encoding.quality)
    else:
        image.save(buffer, format="PNG", optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime_type = encoding.mime_type
    if len(data) > MAX_IMAGE_BASE64_BYTES:
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A") if image.mode == "RGBA" else None)
        buffer = BytesIO()
        background.save(buffer, format="JPEG", quality=85, optimize=True)
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        mime_type = "image/jpeg"
    if len(data) > MAX_IMAGE_BASE64_BYTES:
        raise ValueError(f"computer image exceeds {MAX_IMAGE_BASE64_BYTES:,} base64 bytes")
    return ImageContent(
        type="image",
        mimeType=mime_type,
        data=data,
    )
