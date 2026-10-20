"""Visual rental research through real Chromium, with screenshot-backed grading."""

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastmcp import FastMCP
from hud import Environment
from hud.capabilities import Capability
from hud.environment import Answer
from hud.graders import EvaluationResult
from mcp.types import ContentBlock, ImageContent, TextContent
from playwright.async_api import Browser, Page, Playwright, async_playwright
from pydantic import BaseModel, Field

env = Environment(name="browser-use")
tools = FastMCP("visual-browser")
logger = logging.getLogger(__name__)
URL = "https://diamondrosesanctuary.com/rentals"
PHOTO_IDS = {"69bc30ba4efa7e3bf5b0f5ac_2", "69bc30bab2b1e9ed70de5715_3"}
_page: Page | None = None
_browser: Browser | None = None
_playwright: Playwright | None = None
_server_task: asyncio.Task | None = None
_resources: contextlib.AsyncExitStack | None = None
_observations: dict[str, dict] = {}
_frame_dir = Path("/tmp/visual-research-evidence")


class RentalAnswer(BaseModel):
    chairs: int = Field(ge=0, le=100, description="Number of chairs at the kitchen table")
    evidence_ids: list[str] = Field(
        min_length=1, max_length=5, description="Screenshot IDs observed with the browser tools"
    )
    explanation: str = Field(min_length=10, description="Brief description of the table and how you counted the chairs")


def page() -> Page:
    if _page is None:
        raise RuntimeError("Browser is not initialized")
    return _page


async def observe() -> list[ContentBlock]:
    current = page()
    pixels = await current.screenshot(type="png", animations="disabled")
    images = await current.locator("img").evaluate_all("""images => images.flatMap(img => {
        const r = img.getBoundingClientRect();
        const s = getComputedStyle(img);
        const width = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
        const height = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
        const x = Math.max(0, r.left) + width / 2;
        const y = Math.max(0, r.top) + height / 2;
        const top = document.elementFromPoint(x, y);
        if (!img.complete || !img.naturalWidth || s.visibility !== 'visible' ||
            Number(s.opacity) === 0 || width * height < 150000 ||
            width * height / (r.width * r.height) < 0.75 || top !== img) return [];
        return [{src: img.currentSrc || img.src, visible_area: width * height}];
    })""")
    evidence_id = f"frame-{len(_observations) + 1:03d}"
    _frame_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (_frame_dir / f"{evidence_id}.png").write_bytes(pixels)
    _observations[evidence_id] = {"url": current.url, "images": images}
    visible_text = (await current.locator("body").inner_text())[:18000]
    return [
        TextContent(type="text", text=f"Screenshot ID: {evidence_id}\nURL: {current.url}\n{visible_text}"),
        ImageContent(type="image", data=base64.b64encode(pixels).decode(), mimeType="image/png"),
    ]


@tools.tool
async def screenshot() -> list[ContentBlock]:
    """Inspect the visible browser page and receive a screenshot evidence ID."""
    return await observe()


@tools.tool
async def navigate(url: str) -> list[ContentBlock]:
    """Navigate within the official diamondrosesanctuary.com website."""
    target = urlsplit(url)
    if target.scheme != "https" or target.hostname not in {"diamondrosesanctuary.com", "www.diamondrosesanctuary.com"}:
        raise ValueError("Stay on the official Diamond Rose Sanctuary HTTPS website")
    await page().goto(url, wait_until="domcontentloaded", timeout=45000)
    await page().wait_for_timeout(700)
    return await observe()


@tools.tool
async def click(x: int, y: int) -> list[ContentBlock]:
    """Click a visible link, image, gallery control, or other page element by screenshot coordinates."""
    await page().mouse.click(x, y)
    await page().wait_for_timeout(700)
    return await observe()


@tools.tool
async def scroll(dy: int, dx: int = 0) -> list[ContentBlock]:
    """Scroll the browser viewport; positive dy goes down, negative dy goes up."""
    await page().mouse.wheel(dx, dy)
    await page().wait_for_timeout(700)
    return await observe()


@tools.tool
async def press_key(key: str) -> list[ContentBlock]:
    """Press a key such as ArrowRight/ArrowLeft for a photo gallery or Escape to close it."""
    await page().keyboard.press(key)
    await page().wait_for_timeout(500)
    return await observe()


@tools.tool
async def wait(seconds: float = 1.0) -> list[ContentBlock]:
    """Wait for an image or page animation to settle, for up to five seconds."""
    if not 0 <= seconds <= 5:
        raise ValueError("Wait must be between zero and five seconds")
    await asyncio.sleep(seconds)
    return await observe()


async def listening(port: int, server_task: asyncio.Task | None = None) -> None:
    deadline = asyncio.get_running_loop().time() + 60
    while asyncio.get_running_loop().time() < deadline:
        if server_task is not None and server_task.done():
            await server_task
            raise RuntimeError("Browser tool server exited during startup")
        with socket.socket() as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        await asyncio.sleep(0.25)
    raise RuntimeError(f"Browser service did not start on port {port}")


async def stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


@env.initialize
async def initialize() -> None:
    global _playwright, _browser, _page, _server_task, _resources
    _resources = contextlib.AsyncExitStack()
    try:
        _playwright = await async_playwright().start()
        _resources.push_async_callback(_playwright.stop)
        browser_env = {
            key: os.environ[key] for key in ("HOME", "PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ
        }
        _browser = await _playwright.chromium.launch(
            headless=True,
            channel="chromium",
            args=["--remote-debugging-address=127.0.0.1", "--remote-debugging-port=9222"],
            env=browser_env,
            timeout=60000,
        )
        _resources.push_async_callback(_browser.close)
        context = await _browser.new_context(viewport={"width": 1280, "height": 900})
        _page = await context.new_page()
        await listening(9222)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(
                tools.http_app(path="/mcp"),
                host="127.0.0.1",
                port=port,
                lifespan="on",
                timeout_graceful_shutdown=5,
                log_level="warning",
            )
        )
        _server_task = asyncio.create_task(server.serve())
        _resources.push_async_callback(stop_server, server, _server_task)
        await listening(port, _server_task)
        env.add_capability(Capability.cdp(name="browser", url="http://127.0.0.1:9222"))
        env.add_capability(Capability.mcp(name="browser-ui", url=f"http://127.0.0.1:{port}/mcp"))
    except BaseException:
        await shutdown()
        raise


@env.shutdown
async def shutdown() -> None:
    global _playwright, _browser, _page, _server_task, _resources
    try:
        if _resources is not None:
            await _resources.aclose()
    finally:
        _playwright = _browser = _page = _server_task = _resources = None


def is_kitchen_photo(evidence: dict) -> bool:
    for image in evidence["images"]:
        url = urlsplit(image["src"])
        stem = re.sub(r"-p-\d+$", "", Path(url.path).stem)
        if (
            url.hostname == "cdn.prod.website-files.com"
            and url.path.startswith("/69b9e9887bfbe1b1a4ed614b/")
            and stem in PHOTO_IDS
        ):
            return True
    return False


@env.template(id="rental-kitchen-chairs", returns=RentalAnswer)
async def rental_kitchen_chairs():
    """Inspect rental photos to determine kitchen-table seating, citing recorded visual evidence."""
    _observations.clear()
    await page().goto(URL, wait_until="domcontentloaded", timeout=45000)
    await page().wait_for_timeout(1000)
    response = yield (
        "I'm planning a stay at Diamond Rose Sanctuary. On its official rental page "
        f"({URL}), inspect the kitchen photos and tell me exactly how many chairs are at "
        "the kitchen table. I mean the table in the kitchen with the black-and-white tiled floor, "
        "not the separate formal dining room with white chair covers. Open and inspect the photos "
        "rather than inferring seating from the guest capacity. Do not make a booking or submit forms.\n\n"
        "Browser tools return screenshots with evidence IDs. Use screenshot(), scroll(), click(), "
        "and the gallery arrow keys to inspect the relevant photos. In your final answer, return JSON "
        "with chairs (integer), evidence_ids (one or more screenshot IDs showing the kitchen table), "
        "and explanation (a brief description of how you counted). Return JSON only, without Markdown fences."
    )
    answer = response.content if isinstance(response, Answer) else response
    correct_count = isinstance(answer, RentalAnswer) and answer.chairs == 7
    cited_ids = answer.evidence_ids if isinstance(answer, RentalAnswer) else []
    valid_ids = bool(cited_ids) and all(frame in _observations for frame in cited_ids)
    kitchen_evidence = valid_ids and any(is_kitchen_photo(_observations[frame]) for frame in cited_ids)
    report = {
        "correct_count": correct_count,
        "valid_evidence_ids": valid_ids,
        "kitchen_photo_observed": kitchen_evidence,
        "cited_evidence_ids": cited_ids,
        "observations": _observations,
    }
    logger.info("Visual research grade: %s", json.dumps(report))
    yield EvaluationResult(
        reward=1.0 if correct_count and kitchen_evidence else 0.0,
        content="Exact chair count with recorded kitchen-photo evidence",
        info=report,
    )
