"""A shared desktop serves CDP, RFB and screenshot-evidence tasks."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from hud.environment.server import TaskRunner


async def test_cdp_connection_failure_releases_playwright(browser_env, monkeypatch):
    driver = SimpleNamespace(
        chromium=SimpleNamespace(connect_over_cdp=AsyncMock(side_effect=RuntimeError("CDP connection failed"))),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(browser_env, "wait_for_desktop", AsyncMock())
    monkeypatch.setattr(browser_env, "async_playwright", lambda: SimpleNamespace(start=AsyncMock(return_value=driver)))
    with pytest.raises(RuntimeError, match="CDP connection failed"):
        await browser_env.initialize()
    driver.stop.assert_awaited_once()
    with pytest.raises(RuntimeError, match="Browser is not initialized"):
        browser_env.page()


async def test_desktop_and_mcp_use_one_existing_browser(started_browser):
    service = started_browser
    assert service.env.page() is service.page
    service.driver.chromium.connect_over_cdp.assert_awaited_once_with("http://127.0.0.1:9222")
    assert {cap.protocol for cap in service.env.env.capabilities} == {"cdp/1.3", "rfb/3.8", "mcp/2025-11-25"}
    async with Client(service.env.tools) as client:
        result = await client.call_tool("screenshot")
    assert "frame-001" in next(item.text for item in result.content if item.type == "text")
    service.page.screenshot.assert_awaited_once()
    await service.env.shutdown()
    service.browser.close.assert_awaited_once()
    service.driver.stop.assert_awaited_once()
    assert service.server.should_exit


async def test_rental_scope_does_not_restrict_later_research(started_browser):
    service = started_browser
    runner = TaskRunner(service.env.env.tasks["rental-kitchen-chairs"])
    await runner.start()
    async with Client(service.env.tools) as client:
        result = await client.call_tool("navigate", {"url": "https://example.test/"}, raise_on_error=False)
        assert result.is_error
        assert "official Diamond Rose" in result.content[0].text
    await runner.cancel()
    research = TaskRunner(
        service.env.env.tasks["web-research"],
        {"prompt": "Find the monitoring site", "url": "http://research.test/", "patterns": ["leeds"]},
    )
    await research.start()
    async with Client(service.env.tools) as client:
        result = await client.call_tool("navigate", {"url": "https://example.test/"})
        assert not result.is_error
    assert service.page.url == "https://example.test/"
    assert (await research.grade({"answer": "LEEDS"}))["score"] == 1


@pytest.mark.parametrize("observed", [True, False])
async def test_rental_requires_observed_kitchen_evidence(started_browser, observed):
    service = started_browser
    service.page.images.evaluate_all.return_value = [
        {
            "src": "https://cdn.prod.website-files.com/69b9e9887bfbe1b1a4ed614b/69bc30ba4efa7e3bf5b0f5ac_2.webp",
            "visible_area": 200000,
        }
    ]
    runner = TaskRunner(service.env.env.tasks["rental-kitchen-chairs"])
    await runner.start()
    if observed:
        async with Client(service.env.tools) as client:
            await client.call_tool("screenshot")
    result = await runner.grade(
        {"answer": {"chairs": 7, "evidence_ids": ["frame-001"], "explanation": "Seven chairs at the kitchen table"}}
    )
    assert result["score"] == float(observed)


def test_all_task_rows_bind_to_the_canonical_environment(browser_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "env", browser_env)
    source = Path(__file__).resolve().parents[1] / "tasks.py"
    spec = importlib.util.spec_from_file_location("browser_test_tasks", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert [task.slug for task in module.tasks] == [
        "rental-kitchen-chairs",
        "diamond-rose-seats",
        "epa-air-quality",
        "ladybird-spot-num",
    ]
    assert all(task.env == "browser-use" for task in module.tasks)
