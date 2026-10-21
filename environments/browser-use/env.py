"""Live web research and screenshot-backed tasks in one Chromium desktop."""

import asyncio
import base64
import contextlib
import json
import logging
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastmcp import FastMCP
from hud import Environment
from hud.capabilities import Capability
from hud.environment import Answer
from hud.graders import EvaluationResult, SubScore
from mcp.types import ContentBlock, ImageContent, TextContent
from playwright.async_api import Browser, BrowserContext, Frame, Page, Playwright, async_playwright
from pydantic import BaseModel, Field

from grader import SCREENSHOT_DIR, build_regex_pattern, grade_with_regex, grade_with_rubric

env = Environment(name="browser-use")
env.workspace("/workspace", shell_uid=1000, network=True, require_isolation=True, track_files=True)
tools = FastMCP("visual-browser")
logger = logging.getLogger(__name__)
URL = "https://diamondrosesanctuary.com/rentals"
PHOTO_IDS = {"69bc30ba4efa7e3bf5b0f5ac_2", "69bc30bab2b1e9ed70de5715_3"}
CDP_URL = "http://127.0.0.1:9222"
VNC_HOST = "127.0.0.1"
VNC_PORT = 5900
_browser: Browser | None = None
_playwright: Playwright | None = None
_server_task: asyncio.Task | None = None
_resources: contextlib.AsyncExitStack | None = None
_observations: dict[str, dict] = {}
_visited_domains: list[str] = []
_frame_dir = Path("/tmp/visual-research-evidence")
_rental_task_active = False


class RentalAnswer(BaseModel):
    chairs: int = Field(ge=0, le=100, description="Number of chairs at the kitchen table")
    evidence_ids: list[str] = Field(
        min_length=1, max_length=5, description="Screenshot IDs observed with the browser tools"
    )
    explanation: str = Field(min_length=10, description="Brief description of the table and how you counted the chairs")


def page() -> Page:
    if _browser is None or not _browser.contexts or not _browser.contexts[0].pages:
        raise RuntimeError("Browser is not initialized")
    return _browser.contexts[0].pages[-1]


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
    """Navigate to an HTTP(S) page; the rental task stays on its official website."""
    target = urlsplit(url)
    if target.scheme not in {"http", "https"} or not target.hostname:
        raise ValueError("Browser navigation requires an HTTP or HTTPS URL")
    if _rental_task_active and (
        target.scheme != "https" or target.hostname not in {"diamondrosesanctuary.com", "www.diamondrosesanctuary.com"}
    ):
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


async def wait_for_desktop(timeout: float = 40.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    async with httpx.AsyncClient(timeout=2.0) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                response = await client.get(f"{CDP_URL}/json/version")
                response.raise_for_status()
                if not response.json().get("webSocketDebuggerUrl"):
                    raise ValueError("Chromium did not advertise a CDP websocket")
                reader, writer = await asyncio.wait_for(asyncio.open_connection(VNC_HOST, VNC_PORT), 2)
                try:
                    banner = await asyncio.wait_for(reader.readexactly(12), 2)
                    if not banner.startswith(b"RFB "):
                        raise ValueError("Desktop port did not return an RFB banner")
                finally:
                    writer.close()
                    await writer.wait_closed()
                return
            except (httpx.HTTPError, OSError, TimeoutError, asyncio.IncompleteReadError):
                await asyncio.sleep(0.2)
    raise RuntimeError("Chromium CDP and the RFB desktop did not become ready")


def record_navigation(frame: Frame) -> None:
    if frame != frame.page.main_frame:
        return
    domain = urlsplit(frame.url).netloc
    if domain and domain not in _visited_domains:
        _visited_domains.append(domain)


def watch_page(page: Page) -> None:
    page.on("framenavigated", record_navigation)


def watch_context(context: BrowserContext) -> None:
    context.on("page", watch_page)
    for page in context.pages:
        watch_page(page)


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
    global _playwright, _browser, _server_task, _resources
    await wait_for_desktop()
    _resources = contextlib.AsyncExitStack()
    try:
        _playwright = await async_playwright().start()
        _resources.push_async_callback(_playwright.stop)
        _browser = await _playwright.chromium.connect_over_cdp(CDP_URL)
        _resources.push_async_callback(_browser.close)
        watch_context(_browser.contexts[0])
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
        env.add_capability(Capability.cdp(name="browser", url=CDP_URL))
        env.add_capability(Capability.rfb(name="screen", url=f"rfb://{VNC_HOST}:{VNC_PORT}", display=1))
        env.add_capability(Capability.mcp(name="browser-ui", url=f"http://127.0.0.1:{port}/mcp"))
    except BaseException:
        await shutdown()
        raise


@env.shutdown
async def shutdown() -> None:
    global _playwright, _browser, _server_task, _resources
    try:
        if _resources is not None:
            await _resources.aclose()
    finally:
        _playwright = _browser = _server_task = _resources = None


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
    global _rental_task_active
    _observations.clear()
    _rental_task_active = True
    try:
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
    finally:
        _rental_task_active = False
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


async def capture_screenshots() -> None:
    directory = Path(SCREENSHOT_DIR)
    index = 0
    while True:
        await page().screenshot(path=str(directory / f"frame-{index % 10:02d}.png"))
        index += 1
        await asyncio.sleep(2)


@env.template(id="web-research", description="Answer a research question from a live web page.")
async def web_research(
    prompt: str,
    url: str,
    archives: list[str] | None = None,
    rubric_items: list[dict] | None = None,
    patterns: list[str] | str | None = None,
    use_visual_grading: bool = False,
    hud_api_key: str | None = None,
):
    """Navigate to a starting URL and grade the final answer with a rubric or regex.

    ``hud_api_key`` opts hosted rollouts into protected grader credential injection.
    Archive replay belongs to the older LiveWeb runtime and is not supported here.
    """
    if archives:
        raise ValueError("This LiveWeb template uses the live web; archive replay is not supported")
    target = urlsplit(url)
    if target.scheme not in {"http", "https"} or not target.hostname:
        raise ValueError("Research tasks require an HTTP or HTTPS starting URL")
    _visited_domains.clear()
    current = page()
    await current.goto(url, wait_until="domcontentloaded", timeout=90_000)
    evidence = Path(SCREENSHOT_DIR)
    evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
    for frame in evidence.glob("*.png"):
        frame.unlink()
    capture = asyncio.create_task(capture_screenshots()) if use_visual_grading else None
    try:
        response = yield prompt
    finally:
        if capture is not None:
            capture.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await capture
    answer = response.content if isinstance(response, Answer) else response
    answer = str(answer or "")
    if patterns:
        regex = build_regex_pattern(patterns) if isinstance(patterns, list) else patterns
        yield await grade_with_regex(answer, regex)
        return
    if rubric_items:
        if use_visual_grading:
            await page().screenshot(path=str(evidence / "final.png"))
        result = await grade_with_rubric(
            answer=answer,
            rubric_items=rubric_items,
            query=prompt,
            use_visual_grading=use_visual_grading,
            visited_domains=_visited_domains or None,
        )
        yield EvaluationResult(
            reward=result["score"],
            content=f"Rubric score: {result['score']}",
            subscores=[
                SubScore(name=item["criterion"], weight=item["weight"], value=item["score"])
                for item in result["report"]
            ],
            info={
                "rubric_items": rubric_items,
                "rubric_report": result["report"],
                "subscores": result["subscores"],
                "weights": result["weights"],
                "visited_domains": list(_visited_domains),
            },
        )
        return
    yield 0.0
