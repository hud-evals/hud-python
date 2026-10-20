"""Check LiveWeb task grading and real CDP/RFB readiness contracts offline."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from hud.environment.server import TaskRunner

ROOT = Path(__file__).parents[1]


@pytest.fixture
def liveweb(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVEWEB_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.syspath_prepend(str(ROOT))
    # The examples are standalone projects with identically named local modules.
    monkeypatch.delitem(sys.modules, "grader", raising=False)
    spec = importlib.util.spec_from_file_location("liveweb_test_env", ROOT / "env.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_regex_uses_existing_case_insensitive_alternatives(liveweb):
    import grader

    pattern = grader.build_regex_pattern([r"(?i)leeds", r"(?m)^Birmingham$"])
    assert await grader.grade_with_regex("LEEDS", pattern) == 1
    assert await grader.grade_with_regex("unrelated", pattern) == 0


async def test_rubric_uses_real_weighting_and_records_each_judgment(liveweb, monkeypatch):
    import grader

    create = AsyncMock(
        side_effect=[
            SimpleNamespace(content=[SimpleNamespace(text='{"criterion_status":"MET","explanation":"found"}')]),
            SimpleNamespace(content=[SimpleNamespace(text='{"criterion_status":"UNMET","explanation":"missing"}')]),
        ]
    )
    monkeypatch.setattr(
        grader, "AsyncAnthropic", lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(grader.settings, "api_key", "local-mocked-key")
    result = await grader.grade_with_rubric(
        "Leeds",
        [{"requirement": "Names Leeds", "weight": 20}, {"requirement": "Reports the AQI", "weight": 80}],
        "Find the site and AQI",
        visited_domains=["www.epa.gov"],
    )
    assert result["score"] == 0.2
    assert result["subscores"] == {"criterion_1": 1.0, "criterion_2": 0.0}
    assert result["weights"] == {"criterion_1": 0.2, "criterion_2": 0.8}
    assert [item["verdict"] for item in result["report"]] == ["MET", "UNMET"]
    assert create.await_count == 2
    assert all(call.kwargs["model"] == "claude-opus-4-5" for call in create.await_args_list)
    assert all("www.epa.gov" in call.kwargs["messages"][0]["content"] for call in create.await_args_list)


async def test_task_navigates_and_grades_final_answer_through_runner(liveweb, monkeypatch):
    page = SimpleNamespace(goto=AsyncMock())
    monkeypatch.setattr(liveweb, "browser_page", lambda: page)
    runner = TaskRunner(
        liveweb.env.tasks["web-research"],
        {"prompt": "Find the monitoring site", "url": "https://example.test/", "patterns": ["leeds"]},
    )
    assert await runner.start() == {"prompt": "Find the monitoring site"}
    result = await runner.grade({"answer": "LEEDS"})
    assert result["score"] == 1
    page.goto.assert_awaited_once_with("https://example.test/", wait_until="domcontentloaded", timeout=90_000)


async def test_archive_replay_fails_before_starting_a_research_task(liveweb):
    runner = TaskRunner(
        liveweb.env.tasks["web-research"],
        {"prompt": "Research", "url": "https://example.test/", "archives": ["old-capture"]},
    )
    with pytest.raises(ValueError, match="archive replay is not supported"):
        await runner.start()


async def test_startup_checks_both_cdp_and_rfb_before_publishing(liveweb, monkeypatch):
    requests = []

    def version(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={"webSocketDebuggerUrl": "ws://localhost/browser/test"})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        liveweb.httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(version))
    )
    reader = SimpleNamespace(readexactly=AsyncMock(return_value=b"RFB 003.008\n"))
    writer = SimpleNamespace(close=lambda: None, wait_closed=AsyncMock())
    connection = AsyncMock(return_value=(reader, writer))
    monkeypatch.setattr(liveweb.asyncio, "open_connection", connection)
    context = SimpleNamespace(pages=[], on=lambda *args: None)
    browser = SimpleNamespace(contexts=[context], close=AsyncMock())
    chromium = SimpleNamespace(connect_over_cdp=AsyncMock(return_value=browser))
    playwright = SimpleNamespace(chromium=chromium, stop=AsyncMock())
    monkeypatch.setattr(liveweb, "async_playwright", lambda: SimpleNamespace(start=AsyncMock(return_value=playwright)))
    await liveweb.initialize()
    try:
        assert requests == ["http://127.0.0.1:9222/json/version"]
        connection.assert_awaited_once_with("127.0.0.1", 5900)
        chromium.connect_over_cdp.assert_awaited_once_with("http://127.0.0.1:9222")
        assert {cap.protocol for cap in liveweb.env.capabilities} == {"cdp/1.3", "rfb/3.8"}
    finally:
        await liveweb.shutdown()
    browser.close.assert_awaited_once()
    playwright.stop.assert_awaited_once()


async def test_an_open_non_rfb_port_is_not_a_ready_desktop(liveweb, monkeypatch):
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        liveweb.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"webSocketDebuggerUrl": "ws://localhost/browser/test"})
            )
        ),
    )
    reader = SimpleNamespace(readexactly=AsyncMock(return_value=b"HTTP/1.1 200"))
    writer = SimpleNamespace(close=lambda: None, wait_closed=AsyncMock())
    monkeypatch.setattr(liveweb.asyncio, "open_connection", AsyncMock(return_value=(reader, writer)))
    with pytest.raises(ValueError, match="RFB banner"):
        await liveweb.wait_for_desktop()
    assert liveweb.env.capabilities == []


def test_canonical_tasks_preserve_all_three_existing_slugs():
    tasks = json.loads((ROOT / "tasks.json").read_text())
    assert [task["slug"] for task in tasks] == ["diamond-rose-seats", "epa-air-quality", "ladybird-spot-num"]
    assert {task["id"] for task in tasks} == {"web-research"}
    assert {task["env"] for task in tasks} == {"liveweb"}
