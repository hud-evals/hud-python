"""A deterministic animation coding task with independently rendered physics grading."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import socket
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

import uvicorn
from fastmcp import FastMCP
from hud import Environment
from hud.capabilities import Capability
from hud.graders import EvaluationResult
from mcp.types import ContentBlock, ImageContent, TextContent
from PIL import Image, ImageDraw
from playwright.async_api import Browser, Page, Route, async_playwright
from pydantic import Field
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response

from grader import grade_scene
from motion import capture_motion

ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = (ROOT / "starter" / "README.md").read_text(encoding="utf-8")
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", tempfile.gettempdir() + "/hud-rocket-workspace"))
env = Environment(name="rocket-animation")
env.workspace(WORKSPACE, shell_uid=1000, network=False, track_files=True)
tools = FastMCP("rocket-preview")
_resources: contextlib.AsyncExitStack | None = None
_browser: Browser | None = None
_origin = ""
_preview_count = 0
_preview_source_hash: str | None = None
_motion_source_hash: str | None = None
_lock = asyncio.Lock()


def scene_source() -> str:
    path = WORKSPACE / "scene.js"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Submit a regular scene.js file in the workspace")
    if path.stat().st_size > 128_000:
        raise ValueError("scene.js must be at most 128000 bytes")
    return path.read_text(encoding="utf-8")


async def viewer_route(_: Request) -> Response:
    return HTMLResponse((ROOT / "viewer.html").read_text())


async def scene_route(_: Request) -> Response:
    try:
        source = scene_source()
    except (ValueError, OSError) as exc:
        return Response(str(exc), status_code=400)
    return Response(source, media_type="text/javascript", headers={"Cache-Control": "no-store"})


@contextlib.asynccontextmanager
async def rendered_scene(source: str | None = None) -> AsyncIterator[Page]:
    if _browser is None:
        raise RuntimeError("The preview browser has not started")
    source = scene_source() if source is None else source
    async with await _browser.new_context(viewport={"width": 1280, "height": 960}) as context:

        async def local_only(route: Route) -> None:
            parsed = urlsplit(route.request.url)
            if route.request.url == _origin + "/scene.js":
                await route.fulfill(body=source, content_type="text/javascript")
            elif route.request.url.startswith(_origin + "/") and parsed.path == "/":
                await route.continue_()
            else:
                await route.abort()

        await context.route("**/*", local_only)
        page = await context.new_page()
        page.set_default_timeout(10_000)
        await page.goto(_origin, wait_until="load", timeout=15_000)
        ready = await page.evaluate(
            "async () => { try { await window.rocketViewer.ready; return {ok: true}; } "
            "catch (error) { return {ok: false, error: String(error)}; } }"
        )
        if not ready["ok"]:
            raise ValueError(ready["error"])
        yield page


async def seek(page: Page, time_seconds: float) -> dict:
    response = await asyncio.wait_for(
        page.evaluate(
            "async t => { try { return {ok: true, state: await window.rocketViewer.seek(t)}; } "
            "catch (error) { return {ok: false, error: String(error)}; } }",
            time_seconds,
        ),
        timeout=8,
    )
    if not response["ok"]:
        raise ValueError(response["error"])
    return response["state"]


@tools.tool
async def preview(time_seconds: float = 6.0) -> list[ContentBlock]:
    """Reload scene.js, inspect a frame at 0..18 seconds, and save replay HTML and a frame sheet."""
    global _preview_count, _preview_source_hash
    if not 0 <= time_seconds <= 18:
        raise ValueError("time_seconds must be between 0 and 18")
    source = scene_source()
    async with _lock, rendered_scene(source) as page:
        state = await seek(page, time_seconds)
        pixels = await page.screenshot(type="png")
        await capture_evidence(page)
        export_html(source)
        _preview_source_hash = hashlib.sha256(source.encode()).hexdigest()
        _preview_count += 1
    summary = {
        "time_seconds": time_seconds,
        "rocket": state["rocket"],
        "smoke_particles": len(state["smoke"]),
        "debris_pieces": len(state["debris"]),
        "artifacts": ["rocket-preview.html", "rocket-sequence.png"],
    }
    return [
        TextContent(type="text", text=json.dumps(summary)),
        ImageContent(type="image", data=base64.b64encode(pixels).decode(), mimeType="image/png"),
    ]


async def stop_server(server: uvicorn.Server, task: asyncio.Task) -> None:
    server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


@tools.tool
async def save_replay() -> list[ContentBlock]:
    """Save the finished simulation as an animated WebP, a frame sheet, and an interactive HTML replay."""
    global _preview_source_hash, _motion_source_hash
    source = scene_source()
    async with _lock, rendered_scene(source) as page:
        await capture_motion(page, WORKSPACE / "rocket-animation.webp")
        await capture_evidence(page)
        export_html(source)
        _motion_source_hash = _preview_source_hash = hashlib.sha256(source.encode()).hexdigest()
    return [
        TextContent(
            type="text",
            text="Saved rocket-animation.webp (plays in the trace Workspace), "
            "rocket-sequence.png, and rocket-preview.html (download for interactive playback).",
        ),
        ImageContent(
            type="image",
            data=base64.b64encode((WORKSPACE / "rocket-sequence.png").read_bytes()).decode(),
            mimeType="image/png",
        ),
    ]


@env.initialize
async def initialize() -> None:
    global _resources, _browser, _origin
    resources = _resources = contextlib.AsyncExitStack()
    try:
        playwright = await async_playwright().start()
        resources.push_async_callback(playwright.stop)
        _browser = await playwright.chromium.launch(
            headless=True,
            env={key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR") if key in os.environ},
        )
        resources.push_async_callback(_browser.close)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        _origin = f"http://127.0.0.1:{port}"
        app = tools.http_app(path="/mcp")
        app.add_route("/", viewer_route)
        app.add_route("/scene.js", scene_route)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        task = asyncio.create_task(server.serve())
        resources.push_async_callback(stop_server, server, task)
        async with asyncio.timeout(20):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Preview server exited during startup")
                await asyncio.sleep(0.1)
        env.add_capability(Capability.mcp(name="animation", url=f"{_origin}/mcp"))
    except BaseException:
        await shutdown()
        raise


@env.shutdown
async def shutdown() -> None:
    global _resources, _browser
    if _resources is not None:
        await _resources.aclose()
        _resources = None
        _browser = None


def stage(prompt: str = DEFAULT_PROMPT) -> None:
    global _preview_count, _preview_source_hash, _motion_source_hash
    _preview_count = 0
    _preview_source_hash = None
    _motion_source_hash = None
    for path in WORKSPACE.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    for source in (ROOT / "starter").iterdir():
        target = WORKSPACE / source.name
        shutil.copyfile(source, target)
        if os.geteuid() == 0:
            os.chown(target, 1000, 1000)
    (WORKSPACE / "README.md").write_text(prompt, encoding="utf-8")


def export_html(source: str) -> None:
    module_url = "data:text/javascript;base64," + base64.b64encode(source.encode()).decode()
    viewer = (ROOT / "viewer.html").read_text()
    embedded = f"const SCENE_MODULE_URL = {json.dumps(module_url)};"
    (WORKSPACE / "rocket-preview.html").write_text(viewer.replace("const SCENE_MODULE_URL = '/scene.js';", embedded))


async def capture_evidence(page: Page) -> None:
    frames = []
    for t in (0.0, 1.5, 4.0, 6.0, 7.95, 8.3, 10.0, 12.5, 15.0):
        await seek(page, t)
        pixels = await page.locator(".stage").screenshot(type="png")
        frame = Image.open(io.BytesIO(pixels)).convert("RGB").resize((640, 360))
        ImageDraw.Draw(frame).text((14, 336), f"T + {t:.2f} s", fill="#ffffff")
        frames.append(frame)
    sheet = Image.new("RGB", (1920, 1080), "#080d16")
    for i, frame in enumerate(frames):
        sheet.paste(frame, ((i % 3) * 640, (i // 3) * 360))
    sheet.save(WORKSPACE / "rocket-sequence.png")


@env.template(id="rocket-launch-and-failure")
async def rocket_launch_and_failure(
    prompt: Annotated[str, Field(min_length=1, json_schema_extra={"x-hud-hint": "prompt"})] = DEFAULT_PROMPT,
) -> AsyncIterator[str | EvaluationResult]:
    """Build a rocket's staged launch, max-Q shake, and momentum-preserving mid-air breakup."""
    stage(prompt)
    yield prompt
    async with _lock:
        try:
            source = scene_source()
            async with rendered_scene(source) as page:
                result = await grade_scene(lambda t: seek(page, t))
                await capture_evidence(page)
                export_html(source)
        except ValueError as exc:
            yield EvaluationResult(reward=0, info={"error": str(exc)}, isError=False)
            return
    result.info["preview_calls"] = _preview_count
    result.info["scene_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    result.info["preview_matches_submission"] = _preview_source_hash == result.info["scene_sha256"]
    result.info["animated_replay_matches_submission"] = _motion_source_hash == result.info["scene_sha256"]
    result.info["artifacts"] = ["rocket-preview.html", "rocket-sequence.png"]
    if (WORKSPACE / "rocket-animation.webp").is_file():
        result.info["artifacts"].append("rocket-animation.webp")
    (WORKSPACE / "grading.json").write_text(result.model_dump_json(indent=2))
    yield result
