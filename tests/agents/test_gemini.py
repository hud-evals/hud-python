"""What only ``GeminiAgent`` does: the grounding tool citations add, how it bounds a long
tool output, drops old screenshots and clears a field, how its replies fail, and the
grounding it records as citations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP

from hud.agents import GeminiAgent
from hud.agents.types import GeminiConfig
from hud.capabilities import Capability
from tests.conftest import steps
from tests.fixtures.asgi import mcp_server
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, call
from tests.fixtures.screen import fake_screen
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "gemini-3.1-pro-preview"
CONTROL_L = 0xFFE3


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


async def test_citations_add_google_search_to_the_tools(models: Models, tmp_path: Path) -> None:
    models.script([Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    await task().run(GeminiAgent(GeminiConfig(model=MODEL, citations_enabled=True)))

    (request,) = models.requests()
    assert [t for t in request.body["tools"] if "functionDeclarations" not in t] == [
        {"googleSearch": {}}
    ]


async def test_a_long_tool_output_is_sent_back_with_its_head_and_tail(
    models: Models, tmp_path: Path
) -> None:
    server = FastMCP("long")
    server.tool(name="long_output", description="Return a long text.")(
        lambda: "h" * 500 + "t" * 500
    )
    models.script([call("long_output"), Turn(text="done")])

    async with mcp_server(server) as long_output:
        task = workspace_task(tmp_path / "ws", capabilities=(long_output,))
        await task().run(GeminiAgent(GeminiConfig(model=MODEL, max_tool_result_chars=100)))

    (result,) = models.requests()[-1].tool_results
    output = json.loads(result)["output"]
    assert len(output) < 500
    assert "h" * 10 in output and output.endswith("t" * 10)


async def test_only_the_three_newest_computer_turns_keep_their_screenshots(
    models: Models, tmp_path: Path
) -> None:
    models.script([call("click_at", x=156, y=208)] * 4 + [Turn(text="done")])

    async with fake_screen() as screen:
        task = workspace_task(tmp_path / "ws", capabilities=(Capability.rfb(url=screen.url),))
        await task().run(GeminiAgent(GeminiConfig(model=MODEL)))

    responses = [
        part["functionResponse"]
        for content in models.requests()[-1].body["contents"]
        for part in content["parts"]
        if "functionResponse" in part
    ]
    assert [bool(response.get("parts")) for response in responses] == [False, True, True, True]


async def test_a_field_is_cleared_with_ctrl_a_whatever_system_the_agent_runs_on(
    models: Models, tmp_path: Path
) -> None:
    models.script([call("type_text_at", x=500, y=500, text="hi"), Turn(text="done")])

    async with fake_screen() as screen:
        task = workspace_task(tmp_path / "ws", capabilities=(Capability.rfb(url=screen.url),))
        await task().run(GeminiAgent(GeminiConfig(model=MODEL)))

    pressed = [event.key for event in screen.key_events() if event.down]
    assert pressed[:2] == [CONTROL_L, ord("a")]


@pytest.mark.parametrize(
    ("turn", "error"),
    [
        (Turn(fail=Reply(json={"candidates": []})), "Gemini returned no candidates"),
        (Turn(finish="MALFORMED_FUNCTION_CALL"), "Provider returned a malformed function call"),
    ],
    ids=["no-candidates", "a-malformed-function-call"],
)
async def test_a_reply_it_cannot_use_ends_the_run_with_its_error(
    turn: Turn, error: str, models: Models, tmp_path: Path
) -> None:
    models.queue([turn])

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(GeminiAgent(GeminiConfig(model=MODEL)))).runs

    assert (run.trace.status, len(models.requests())) == ("error", 1)
    assert error in (run.trace.error or "")


async def test_grounding_and_thoughts_land_on_the_agent_step(
    models: Models, tmp_path: Path
) -> None:
    web = "https://example.com/sky"
    grounding = {
        "groundingChunks": [
            {"web": {"uri": web, "title": "Sky"}},
            {"web": {"uri": "https://example.com/unused", "title": "Unused"}},
        ],
        "groundingSupports": [
            {
                "segment": {"text": "Blue.", "startIndex": 0, "endIndex": 5},
                "groundingChunkIndices": [0],
            }
        ],
    }
    models.script(
        [Turn(text="Blue.", reasoning="look it up", extra={"groundingMetadata": grounding})]
    )

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(GeminiAgent(GeminiConfig(model=MODEL)))).runs

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert (step["content"], step["reasoning"]) == ("Blue.", "look it up")
    assert [(c["type"], c["source"], c["text"]) for c in step["citations"]] == [
        ("grounding", web, "Blue."),
        ("grounding", "https://example.com/unused", ""),
    ]
