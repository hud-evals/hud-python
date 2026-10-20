"""What only ``OpenAIAgent`` does over the Responses API: the tool search it offers, how it
chains responses and discards an unusable one, how a refused, capped or incomplete call
lands on the trace, and the citations it records."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP

from hud.agents import OpenAIAgent
from hud.agents.openai.tools import OpenAIToolSearchTool
from hud.agents.types import OpenAIConfig
from tests.agents.conftest import tool_results
from tests.conftest import steps
from tests.fixtures.asgi import mcp_server
from tests.fixtures.environments import workspace_task
from tests.fixtures.models import Turn, shell, shell_call

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models

MODEL = "gpt-5.6"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


@pytest.mark.parametrize(
    ("threshold", "deferred"),
    [(1, {"database__lookup", "search__lookup"}), (2, set())],
    ids=["over-its-threshold", "at-its-threshold"],
)
async def test_tool_search_defers_mcp_tools_once_there_are_more_than_its_threshold(
    threshold: int, deferred: set[str], models: Models, tmp_path: Path
) -> None:
    lookups = FastMCP("lookups")
    lookups.tool(name="lookup", description="Look a key up.")(lambda key: key)
    models.script([Turn(text="done")])
    agent = OpenAIAgent(
        OpenAIConfig(model=MODEL, hosted_tools=[OpenAIToolSearchTool(threshold=threshold)])
    )

    async with (
        mcp_server(lookups, name="database") as database,
        mcp_server(lookups, name="search") as search,
    ):
        task = workspace_task(tmp_path / "ws", capabilities=(database, search))
        await task().run(agent)

    (request,) = models.requests()
    assert {tool["name"] for tool in request.body["tools"] if tool.get("defer_loading")} == deferred


async def test_citations_ask_for_web_search_sources(models: Models, tmp_path: Path) -> None:
    models.script([Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    await task().run(OpenAIAgent(OpenAIConfig(model=MODEL, citations_enabled=True)))

    (request,) = models.requests()
    assert request.body["include"] == ["web_search_call.action.sources"]


async def test_each_request_sends_only_what_is_new_since_the_response_it_continues(
    models: Models, tmp_path: Path
) -> None:
    models.script([shell("echo hi"), Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    await task().run(OpenAIAgent(OpenAIConfig(model=MODEL)))

    first, second = (request.body for request in models.requests())
    assert (first.get("previous_response_id"), second["previous_response_id"]) == (None, "resp_1")
    assert [item["type"] for item in second["input"]] == ["shell_call_output"]


async def test_an_empty_shell_call_is_sampled_again_without_advancing_the_chain(
    models: Models, tmp_path: Path
) -> None:
    models.queue([shell_call("true"), shell_call(), Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(OpenAIAgent(OpenAIConfig(model=MODEL)))).runs

    chain = [request.body.get("previous_response_id") for request in models.requests()]
    assert (run.trace.content, chain) == ("done", [None, "resp_1", "resp_1"])


async def test_a_shell_call_it_cannot_run_fails_its_tool_step_and_the_run_goes_on(
    models: Models, tmp_path: Path
) -> None:
    refused = shell_call("true")
    refused = Turn(native=({**refused.native[0], "action": {"commands": 5}},))
    models.script([refused, Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(OpenAIAgent(OpenAIConfig(model=MODEL)))).runs

    ((failed, text),) = tool_results(run)
    assert (run.trace.status, failed) == ("completed", True)
    assert "commands must be a list of strings" in text


async def test_a_shell_calls_output_cap_keeps_the_head_and_tail_of_its_output(
    models: Models, tmp_path: Path
) -> None:
    models.script([shell_call("seq 1 1000", max_output_length=50), Turn(text="done")])

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(OpenAIAgent(OpenAIConfig(model=MODEL)))).runs

    ((_, text),) = tool_results(run)
    assert len(text) <= 50
    assert text.startswith("1\n2\n") and text.endswith("999\n1000\n")


async def test_a_function_call_cut_off_mid_arguments_fails_its_tool_step_unrun(
    models: Models, tmp_path: Path
) -> None:
    lookups = FastMCP("lookups")
    lookups.tool(name="lookup", description="Look a key up.")(lambda key: key)
    cut_off = Turn(
        native=(
            {
                "type": "function_call",
                "id": "fc_cut",
                "call_id": "call_cut",
                "name": "lookup",
                "arguments": '{"key": "ab',
                "status": "completed",
            },
        )
    )
    models.script([cut_off, Turn(text="done")])

    async with mcp_server(lookups) as server:
        task = workspace_task(tmp_path / "ws", capabilities=(server,))
        (run,) = (await task().run(OpenAIAgent(OpenAIConfig(model=MODEL)))).runs

    ((failed, text),) = tool_results(run)
    assert (run.trace.status, failed) == ("completed", True)
    assert "the arguments for this 'lookup' call arrived incomplete" in text


async def test_url_and_file_citations_land_on_the_agent_step(
    models: Models, tmp_path: Path
) -> None:
    web = "https://example.com/sky"
    answer = {
        "type": "message",
        "id": "msg_cited",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Blue.",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url": web,
                        "title": "Sky",
                        "start_index": 0,
                        "end_index": 5,
                    },
                    {
                        "type": "file_citation",
                        "file_id": "file-1",
                        "filename": "notes.txt",
                        "index": 0,
                    },
                ],
            }
        ],
    }
    models.script([Turn(reasoning="look it up", native=(answer,))])

    task = workspace_task(tmp_path / "ws")
    (run,) = (await task().run(OpenAIAgent(OpenAIConfig(model=MODEL)))).runs

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert (step["content"], step["reasoning"]) == ("Blue.", "look it up")
    assert [(c["type"], c["source"], c["text"]) for c in step["citations"]] == [
        ("url_citation", web, "Sky"),
        ("file_citation", "file-1", "notes.txt"),
    ]
