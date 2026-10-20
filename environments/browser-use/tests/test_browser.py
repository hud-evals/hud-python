"""Browser lifecycle and the shared MCP/CDP observation contract."""

import base64
import importlib.util
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp import Client
from playwright.async_api import async_playwright


@pytest.fixture
def browser_env(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "env.py"
    spec = importlib.util.spec_from_file_location("featured_browser_test_env", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_frame_dir", tmp_path / "frames")
    return module


@pytest.mark.asyncio
async def test_browser_launch_failure_releases_playwright(browser_env, monkeypatch):
    driver = SimpleNamespace(
        chromium=SimpleNamespace(launch=AsyncMock(side_effect=RuntimeError("Chromium launch failed"))),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(browser_env, "async_playwright", lambda: SimpleNamespace(start=AsyncMock(return_value=driver)))
    monkeypatch.setenv("HUD_API_KEY", "test-secret-not-for-browser")

    try:
        with pytest.raises(RuntimeError, match="Chromium launch failed"):
            await browser_env.env.start()
        driver.stop.assert_awaited_once()
        assert "HUD_API_KEY" not in driver.chromium.launch.call_args.kwargs["env"]
    finally:
        await browser_env.env.stop()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_and_cdp_share_the_page_and_close_on_stop(browser_env):
    await browser_env.env.start()
    cdp_url = browser_env.env.capability("browser").url
    mcp_url = browser_env.env.capability("browser-ui").url
    try:
        async with async_playwright() as playwright:
            remote = await playwright.chromium.connect_over_cdp(cdp_url)
            page = remote.contexts[0].pages[0]
            await page.goto(
                "data:text/html,<h1>Browser smoke</h1><button onclick=\"this.textContent='Clicked'\">Click me</button>"
            )
            async with Client(mcp_url) as client:
                result = await client.call_tool("screenshot")
                text = next(item.text for item in result.content if item.type == "text")
                pixels = base64.b64decode(next(item.data for item in result.content if item.type == "image"))
                assert "Browser smoke" in text and "frame-001" in text
                assert pixels.startswith(b"\x89PNG\r\n\x1a\n")
                assert struct.unpack(">II", pixels[16:24]) == (1280, 900)
                button = await page.get_by_role("button").bounding_box()
                result = await client.call_tool(
                    "click", {"x": int(button["x"] + button["width"] / 2), "y": int(button["y"] + button["height"] / 2)}
                )
                assert "Clicked" in next(item.text for item in result.content if item.type == "text")
    finally:
        await browser_env.env.stop()

    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        for url in (cdp_url, mcp_url):
            with pytest.raises(httpx.ConnectError):
                await client.get(url)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_server_failure_closes_the_browser(browser_env, monkeypatch):
    async def fail_server(self):
        raise RuntimeError("Tool server startup failed")

    monkeypatch.setattr(browser_env.uvicorn.Server, "serve", fail_server)
    try:
        with pytest.raises(RuntimeError, match="Tool server startup failed"):
            await browser_env.env.start()
        with pytest.raises(RuntimeError, match="Browser is not initialized"):
            browser_env.page()
        async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get("http://127.0.0.1:9222/json/version")
    finally:
        await browser_env.env.stop()
