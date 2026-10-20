"""What only ``OpenAIAgent`` does over the Responses API: the hosted tools and tool search it
offers, how it chains responses and discards an unusable one, how it reports a refused or
incomplete call to the model, and the citations it records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from fastmcp import FastMCP

from hud.agents import OpenAIAgent
from hud.agents.openai.tools import OpenAICodeInterpreterTool, OpenAIToolSearchTool
from hud.agents.types import OpenAIConfig
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Turn, mcp_server, say, shell, shell_call, steps

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, Models

MODEL = "gpt-5.6"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


async def test_a_hosted_code_interpreter_is_offered_as_configured(
    models: Models, tmp_path: Path
) -> None:
    hosted = OpenAICodeInterpreterTool(container={"type": "auto"})
    models.script([say("done")])

    await run_task(
        workspace_env(tmp_path / "ws"),
        OpenAIAgent(OpenAIConfig(model=MODEL, hosted_tools=[hosted])),
    )

    (request,) = models.requests()
    assert [t for t in request.body["tools"] if t["type"] == "code_interpreter"] == [
        {"type": "code_interpreter", "container": {"type": "auto"}}
    ]


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
    models.script([say("done")])
    agent = OpenAIAgent(
        OpenAIConfig(model=MODEL, hosted_tools=[OpenAIToolSearchTool(threshold=threshold)])
    )

    async with (
        mcp_server(lookups, name="database") as database,
        mcp_server(lookups, name="search") as search,
    ):
        await run_task(workspace_env(tmp_path / "ws", capabilities=(database, search)), agent)

    (request,) = models.requests()
    assert {tool["name"] for tool in request.body["tools"] if tool.get("defer_loading")} == deferred


async def test_citations_ask_for_web_search_sources(models: Models, tmp_path: Path) -> None:
    models.script([say("done")])

    await run_task(
        workspace_env(tmp_path / "ws"),
        OpenAIAgent(OpenAIConfig(model=MODEL, citations_enabled=True)),
    )

    (request,) = models.requests()
    assert request.body["include"] == ["web_search_call.action.sources"]


async def test_each_request_sends_only_what_is_new_since_the_response_it_continues(
    models: Models, tmp_path: Path
) -> None:
    models.script([shell("echo hi"), say("done")])

    await run_task(workspace_env(tmp_path / "ws"), OpenAIAgent(OpenAIConfig(model=MODEL)))

    first, second = (request.body for request in models.requests())
    assert (first.get("previous_response_id"), second["previous_response_id"]) == (None, "resp_1")
    assert [item["type"] for item in second["input"]] == ["shell_call_output"]


async def test_an_empty_shell_call_is_sampled_again_without_advancing_the_chain(
    models: Models, tmp_path: Path
) -> None:
    models.queue([shell_call("true"), shell_call(), say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), OpenAIAgent(OpenAIConfig(model=MODEL)))

    chain = [request.body.get("previous_response_id") for request in models.requests()]
    assert (run.trace.content, chain) == ("done", [None, "resp_1", "resp_1"])


async def test_a_shell_call_it_cannot_run_is_told_to_the_model(
    models: Models, tmp_path: Path
) -> None:
    refused = shell_call("true")
    refused = Turn(native=({**refused.native[0], "action": {"commands": 5}},))
    models.script([refused, say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), OpenAIAgent(OpenAIConfig(model=MODEL)))

    assert run.trace.status == "completed"
    assert "commands must be a list of strings" in json.dumps(models.requests()[-1].body)


async def test_a_shell_calls_output_cap_reaches_the_model_with_its_output(
    models: Models, tmp_path: Path
) -> None:
    models.script([shell_call("seq 1 1000", max_output_length=50), say("done")])

    await run_task(workspace_env(tmp_path / "ws"), OpenAIAgent(OpenAIConfig(model=MODEL)))

    (output,) = [i for i in models.requests()[-1].body["input"] if i["type"] == "shell_call_output"]
    assert output["max_output_length"] == 50


async def test_a_function_call_cut_off_mid_arguments_is_explained_to_the_model(
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
    models.script([cut_off, say("done")])

    async with mcp_server(lookups) as server:
        env = workspace_env(tmp_path / "ws", capabilities=(server,))
        run = await run_task(env, OpenAIAgent(OpenAIConfig(model=MODEL)))

    assert run.trace.status == "completed"
    assert "the arguments for this 'lookup' call arrived incomplete" in json.dumps(
        models.requests()[-1].body
    )


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

    run = await run_task(workspace_env(tmp_path / "ws"), OpenAIAgent(OpenAIConfig(model=MODEL)))

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert (step["content"], step["reasoning"]) == ("Blue.", "look it up")
    assert [(c["type"], c["source"], c["text"]) for c in step["citations"]] == [
        ("url_citation", web, "Sky"),
        ("file_citation", "file-1", "notes.txt"),
    ]
