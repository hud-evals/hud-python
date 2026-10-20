"""Research arbitrary live websites through a Linux desktop and Chromium."""

import asyncio
import contextlib
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from hud import Environment
from hud.capabilities import Capability
from hud.environment import Answer
from hud.graders import EvaluationResult, SubScore
from playwright.async_api import Browser, BrowserContext, Frame, Page, Playwright, async_playwright

from grader import SCREENSHOT_DIR, build_regex_pattern, grade_with_regex, grade_with_rubric

env = Environment(name="liveweb")
env.workspace("/workspace", shell_uid=1000, network=True, require_isolation=True, track_files=True)

CDP_URL = "http://127.0.0.1:9222"
VNC_HOST = "127.0.0.1"
VNC_PORT = 5900
_playwright: Playwright | None = None
_browser: Browser | None = None
_visited_domains: list[str] = []


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


@env.initialize
async def initialize() -> None:
    global _playwright, _browser
    await wait_for_desktop()
    _playwright = await async_playwright().start()
    try:
        _browser = await _playwright.chromium.connect_over_cdp(CDP_URL)
        for context in _browser.contexts:
            watch_context(context)
        env.add_capability(Capability.cdp(name="browser", url=CDP_URL))
        env.add_capability(Capability.rfb(name="screen", url=f"rfb://{VNC_HOST}:{VNC_PORT}", display=1))
    except BaseException:
        await shutdown()
        raise


@env.shutdown
async def shutdown() -> None:
    global _playwright, _browser
    try:
        if _browser is not None:
            await _browser.close()
    finally:
        _browser = None
        if _playwright is not None:
            await _playwright.stop()
            _playwright = None


def browser_page() -> Page:
    if _browser is None or not _browser.contexts or not _browser.contexts[0].pages:
        raise RuntimeError("The browser has not started")
    return _browser.contexts[0].pages[-1]


async def capture_screenshots() -> None:
    directory = Path(SCREENSHOT_DIR)
    index = 0
    while True:
        await browser_page().screenshot(path=str(directory / f"frame-{index % 10:02d}.png"))
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
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Research tasks require an HTTP or HTTPS starting URL")
    _visited_domains.clear()
    page = browser_page()
    await page.goto(url, wait_until="domcontentloaded", timeout=90_000)
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
            await browser_page().screenshot(path=str(evidence / "final.png"))
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
