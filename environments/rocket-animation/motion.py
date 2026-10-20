"""Render a watchable motion artifact from the trusted rocket viewer."""

import io
from pathlib import Path

from PIL import Image
from playwright.async_api import Page


async def capture_motion(page: Page, destination: Path) -> None:
    frames: list[Image.Image] = []
    stage = page.locator(".stage")
    for index in range(108):
        timestamp = index / 6
        response = await page.evaluate(
            "async t => { try { await window.rocketViewer.seek(t); return {ok: true}; } "
            "catch (error) { return {ok: false, error: String(error)}; } }",
            timestamp,
        )
        if not response["ok"]:
            raise ValueError(f"Scene failed at {timestamp:.3f}s: {response['error']}")
        pixels = await stage.screenshot(type="png")
        with Image.open(io.BytesIO(pixels)) as image:
            frames.append(image.convert("RGB").resize((640, 360), Image.Resampling.LANCZOS))
    frames[0].save(
        destination,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=167,
        loop=0,
        quality=65,
        method=3,
    )
    if destination.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("Motion artifact exceeds the 5 MiB file-capture limit")
