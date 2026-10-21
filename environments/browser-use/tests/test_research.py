"""Check LiveWeb task grading and real CDP/RFB readiness contracts offline."""

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from hud.environment.server import TaskRunner

ROOT = Path(__file__).parents[1]


async def test_regex_uses_existing_case_insensitive_alternatives(browser_env):
    import grader

    pattern = grader.build_regex_pattern([r"(?i)leeds", r"(?m)^Birmingham$"])
    assert await grader.grade_with_regex("LEEDS", pattern) == 1
    assert await grader.grade_with_regex("unrelated", pattern) == 0


async def test_rubric_uses_real_weighting_and_records_each_judgment(browser_env, monkeypatch):
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
    assert all(
        "<domains_visited>\nwww.epa.gov\n</domains_visited>" in call.kwargs["messages"][0]["content"]
        for call in create.await_args_list
    )


@pytest.mark.parametrize(
    "raw_weights,verdicts,expected_score,display_weights",
    [
        ([1, 1, 1], ["MET", "MET", "MET"], 1.0, [0.333, 0.333, 0.333]),
        ([1, 1, 1], ["MET", "MET", "UNMET"], 0.667, [0.333, 0.333, 0.333]),
        ([1, 1, 1, -1], ["MET"] * 4, 0.667, [0.333, 0.333, 0.333, -0.333]),
        ([1, -1, -1, -5], ["MET"] * 4, 0.1, [1.0, -0.129, -0.129, -0.643]),
    ],
)
async def test_rubric_scores_use_full_precision_before_display_rounding(
    browser_env, monkeypatch, raw_weights, verdicts, expected_score, display_weights
):
    import grader

    create = AsyncMock(
        side_effect=[
            SimpleNamespace(
                content=[SimpleNamespace(text=json.dumps({"criterion_status": verdict, "explanation": "judged"}))]
            )
            for verdict in verdicts
        ]
    )
    monkeypatch.setattr(
        grader, "AsyncAnthropic", lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(grader.settings, "api_key", "local-mocked-key")
    result = await grader.grade_with_rubric(
        "The completed research",
        [{"requirement": f"Criterion {index}", "weight": weight} for index, weight in enumerate(raw_weights)],
        "Research the question",
    )

    assert result["score"] == expected_score
    assert [item["weight"] for item in result["report"]] == display_weights
    assert result["weights"] == {
        f"criterion_{index + 1}": weight for index, weight in enumerate(display_weights) if raw_weights[index] >= 0
    }
    assert [item["verdict"] for item in result["report"]] == verdicts
    assert create.await_count == len(raw_weights)


@pytest.mark.parametrize("limit", [0, 1, 10])
def test_visual_evidence_keeps_final_frame_after_recent_ring_samples(browser_env, limit):
    import grader

    directory = Path(grader.SCREENSHOT_DIR)
    directory.mkdir()
    chronological = [f"frame-{index % 10:02d}.png" for index in range(7, 17)] + ["final.png"]
    for index, name in enumerate(chronological):
        path = directory / name
        path.write_bytes(name.encode())
        os.utime(path, (100 + index, 100 + index))

    images = grader.load_screenshots_for_grading(max_screenshots=limit)
    selected = [base64.b64decode(image["source"]["data"]).decode() for image in images]

    assert selected == (chronological[-limit:] if limit else [])


async def test_task_navigates_and_grades_final_answer_through_runner(started_browser):
    browser_env = started_browser.env
    runner = TaskRunner(
        browser_env.env.tasks["web-research"],
        {"prompt": "Find the monitoring site", "url": "https://example.test/", "patterns": ["leeds"]},
    )
    assert await runner.start() == {"prompt": "Find the monitoring site"}
    result = await runner.grade({"answer": "LEEDS"})
    assert result["score"] == 1
    started_browser.page.goto.assert_awaited_once_with(
        "https://example.test/", wait_until="domcontentloaded", timeout=90_000
    )


async def test_archive_replay_fails_before_starting_a_research_task(browser_env):
    runner = TaskRunner(
        browser_env.env.tasks["web-research"],
        {"prompt": "Research", "url": "https://example.test/", "archives": ["old-capture"]},
    )
    with pytest.raises(ValueError, match="archive replay is not supported"):
        await runner.start()


async def test_startup_checks_both_cdp_and_rfb_before_publishing(browser_env, monkeypatch):
    requests = []

    def version(request):
        requests.append(str(request.url))
        return httpx.Response(200, json={"webSocketDebuggerUrl": "ws://localhost/browser/test"})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        browser_env.httpx, "AsyncClient", lambda **kwargs: original_client(transport=httpx.MockTransport(version))
    )
    reader = SimpleNamespace(readexactly=AsyncMock(return_value=b"RFB 003.008\n"))
    writer = SimpleNamespace(close=lambda: None, wait_closed=AsyncMock())
    connection = AsyncMock(return_value=(reader, writer))
    monkeypatch.setattr(browser_env.asyncio, "open_connection", connection)
    await browser_env.wait_for_desktop()
    assert requests == ["http://127.0.0.1:9222/json/version"]
    connection.assert_awaited_once_with("127.0.0.1", 5900)


async def test_an_open_non_rfb_port_is_not_a_ready_desktop(browser_env, monkeypatch):
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        browser_env.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"webSocketDebuggerUrl": "ws://localhost/browser/test"})
            )
        ),
    )
    reader = SimpleNamespace(readexactly=AsyncMock(return_value=b"HTTP/1.1 200"))
    writer = SimpleNamespace(close=lambda: None, wait_closed=AsyncMock())
    monkeypatch.setattr(browser_env.asyncio, "open_connection", AsyncMock(return_value=(reader, writer)))
    with pytest.raises(ValueError, match="RFB banner"):
        await browser_env.wait_for_desktop()
    assert {cap.protocol for cap in browser_env.env.capabilities} == set()


def test_canonical_tasks_preserve_both_existing_tasksets():
    tasks = json.loads((ROOT / "tasks.json").read_text())
    assert [task["slug"] for task in tasks] == [
        "rental-kitchen-chairs",
        "diamond-rose-seats",
        "epa-air-quality",
        "ladybird-spot-num",
    ]
    assert {task["id"] for task in tasks} == {"rental-kitchen-chairs", "web-research"}
    assert {task["env"] for task in tasks} == {"browser-use"}
