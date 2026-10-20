"""Computer toolset image bounds and native-display coordinate mapping."""

from __future__ import annotations

import base64
import math
import random
from io import BytesIO

import pytest
from PIL import Image

from hud.agents.claude.tools.geometry import ComputerGeometry
from hud.capabilities.rfb import PngScreenshotEncoding, ScreenshotEncoding, WebPScreenshotEncoding


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "size",
    [
        (1280, 800),
        (3840, 2160),
        (2160, 3840),
        (3840, 3840),
        (2576, 1456),
        (100000, 1),
        (1, 100000),
    ],
)
def test_screenshot_size_meets_visual_token_and_history_limits(size: tuple[int, int]) -> None:
    width, height = ComputerGeometry(*size).screenshot_size

    assert max(width, height) <= 2000
    assert math.ceil(width / 28) * math.ceil(height / 28) <= 4784
    assert width <= size[0] and height <= size[1]
    assert abs(width * size[1] - height * size[0]) <= max(size)


def test_square_screenshots_respect_rounded_visual_token_count() -> None:
    geometry = ComputerGeometry(2000, 2000)

    assert geometry.screenshot_size == (1932, 1932)


def test_small_display_preserves_its_coordinate_space() -> None:
    geometry = ComputerGeometry(1280, 800)

    assert geometry.screenshot_size == (1280, 800)
    assert geometry.to_screen(123, 456) == (123, 456)
    assert geometry.from_screen(123, 456) == (123, 456)


def test_scaled_display_maps_actions_and_cursor_in_opposite_directions() -> None:
    geometry = ComputerGeometry(3840, 2160)

    assert geometry.screenshot_size == (2000, 1125)
    assert geometry.to_screen(1000, 500) == (1920, 960)
    assert geometry.from_screen(1920, 960) == (1000, 500)
    assert geometry.from_screen(3839, 2159) == (1999, 1124)


@pytest.mark.parametrize("point", [(-1, 0), (0, -1), (2000, 0), (0, 1125)])
def test_actions_outside_the_screenshot_fail(point: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="outside"):
        ComputerGeometry(3840, 2160).to_screen(*point)


@pytest.mark.parametrize("point", [(-1, 0), (3840, 0), (0, 2160)])
def test_cursor_outside_the_native_display_fails(point: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="outside"):
        ComputerGeometry(3840, 2160).from_screen(*point)


def test_zoom_regions_cover_the_requested_pixels_and_include_display_edges() -> None:
    geometry = ComputerGeometry(3840, 2160)

    assert geometry.screen_region((0, 0, 2000, 1125)) == (0, 0, 3840, 2160)
    assert geometry.screen_region((1, 1, 2, 2)) == (1, 1, 4, 4)


@pytest.mark.parametrize(
    "region",
    [(-1, 0, 100, 100), (0, 0, 2001, 100), (0, 0, 100, 1126), (10, 10, 10, 20)],
)
def test_invalid_zoom_regions_fail(region: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError, match="outside"):
        ComputerGeometry(3840, 2160).screen_region(region)


@pytest.mark.parametrize("encoding", [PngScreenshotEncoding(), WebPScreenshotEncoding(quality=42)])
def test_screenshot_is_encoded_at_the_advertised_geometry(encoding: ScreenshotEncoding) -> None:
    geometry = ComputerGeometry(3840, 2160)
    content = geometry.screenshot(_png(Image.new("RGB", (3840, 2160), "red")), encoding)

    assert content.mimeType == encoding.mime_type
    with Image.open(BytesIO(base64.b64decode(content.data))) as image:
        assert image.size == geometry.screenshot_size
        assert image.format == ("WEBP" if isinstance(encoding, WebPScreenshotEncoding) else "PNG")


def test_zoom_crops_native_pixels_without_changing_future_action_coordinates() -> None:
    geometry = ComputerGeometry(3840, 2160)
    screenshot = Image.new("RGB", (3840, 2160), "blue")
    screenshot.paste("red", (960, 480, 1920, 960))

    content = geometry.zoom(_png(screenshot), (500, 250, 1000, 500), PngScreenshotEncoding())

    with Image.open(BytesIO(base64.b64decode(content.data))) as image:
        assert image.size == (960, 480)
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((959, 479)) == (255, 0, 0)
    assert geometry.to_screen(1000, 500) == (1920, 960)


def test_noisy_png_falls_back_to_jpeg_without_changing_coordinate_space() -> None:
    geometry = ComputerGeometry(2000, 2000)
    screenshot = Image.frombytes("RGB", (2000, 2000), random.Random(0).randbytes(2000 * 2000 * 3))

    content = geometry.screenshot(_png(screenshot), PngScreenshotEncoding())

    assert content.mimeType == "image/jpeg"
    assert len(content.data) <= 5_000_000
    with Image.open(BytesIO(base64.b64decode(content.data))) as image:
        assert image.format == "JPEG"
        assert image.size == geometry.screenshot_size


def test_large_zoom_is_limited_to_the_usual_screenshot_dimensions() -> None:
    geometry = ComputerGeometry(3840, 2160)
    content = geometry.zoom(
        _png(Image.new("RGB", (3840, 2160))), (0, 0, 1500, 1125), PngScreenshotEncoding()
    )

    with Image.open(BytesIO(base64.b64decode(content.data))) as image:
        assert image.size == (1500, 1125)


def test_screenshot_with_changed_display_dimensions_fails() -> None:
    with pytest.raises(ValueError, match="does not match"):
        ComputerGeometry(1280, 800).screenshot(
            _png(Image.new("RGB", (640, 400))), PngScreenshotEncoding()
        )


@pytest.mark.parametrize("size", [(0, 800), (1280, 0), (-1, 800)])
def test_invalid_display_dimensions_fail(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        ComputerGeometry(*size)
