"""What only ``OpenAIAgent`` sends and records: its hosted tools and tool search in the
Responses request, the sources it asks for, how it keeps its response chain when it
discards a turn, and the citations it reads back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import FastMCP

from hud.agents import OpenAIAgent
from hud.agents.openai.tools import OpenAICodeInterpreterTool, OpenAIToolSearchTool
from hud.agents.types import OpenAIConfig
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Turn, mcp_server, say, shell_call, steps

if TYPE_CHECKING:
    from pathlib import Path

    from hud.capabilities import Capability
    from tests.support import HudEnv, ModelRequest, Models


def openai(**config: Any) -> OpenAIAgent:
    return OpenAIAgent(OpenAIConfig.model_validate({"model": "gpt-5.6", **config}))


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


async def first_request(
    models: Models, root: Path, capabilities: tuple[Capability, ...] = (), **config: Any
) -> ModelRequest:
    models.script([say("done")])
    await run_task(workspace_env(root, capabilities=capabilities), openai(**config))
    return models.requests()[0]


def lookups() -> FastMCP:
    server = FastMCP("lookups")
    server.tool(name="lookup", description="Look a key up.")(lambda key: key)
    return server


async def test_a_hosted_code_interpreter_is_offered_as_configured(
    models: Models, tmp_path: Path
) -> None:
    hosted = OpenAICodeInterpreterTool(container={"type": "auto"})

    request = await first_request(models, tmp_path / "ws", hosted_tools=[hosted])

    offered = [tool for tool in request.body["tools"] if tool["type"] == "code_interpreter"]
    assert offered == [{"type": "code_interpreter", "container": {"type": "auto"}}]


@pytest.mark.parametrize(
    ("threshold", "deferred"),
    [(1, {"database__lookup", "search__lookup"}), (5, set())],
    ids=["over-its-threshold", "under-its-threshold"],
)
async def test_tool_search_defers_mcp_tools_once_there_are_more_than_its_threshold(
    threshold: int, deferred: set[str], models: Models, tmp_path: Path
) -> None:
    async with (
        mcp_server(lookups(), name="database") as database,
        mcp_server(lookups(), name="search") as search,
    ):
        request = await first_request(
            models,
            tmp_path / "ws",
            (database, search),
            hosted_tools=[OpenAIToolSearchTool(threshold=threshold)],
        )

    assert {tool["name"] for tool in request.body["tools"] if tool.get("defer_loading")} == deferred


async def test_citations_ask_for_web_search_sources(models: Models, tmp_path: Path) -> None:
    request = await first_request(models, tmp_path / "ws", citations_enabled=True)

    assert request.body.get("include") == ["web_search_call.action.sources"]


async def test_an_empty_shell_call_is_resampled_without_advancing_the_response_chain(
    models: Models, tmp_path: Path
) -> None:
    models.queue([shell_call("true"), shell_call(), say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), openai())

    chain = [request.body.get("previous_response_id") for request in models.requests()]
    assert (run.trace.content, chain) == ("done", [None, "resp_1", "resp_1"])


async def test_url_and_file_citations_land_on_the_agent_step(
    models: Models, tmp_path: Path
) -> None:
    web = "https://example.com/sky"
    models.script(
        [
            Turn(
                reasoning="look it up",
                native=(
                    {
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
                    },
                ),
            )
        ]
    )

    run = await run_task(workspace_env(tmp_path / "ws"), openai())

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert (step["content"], step["reasoning"]) == ("Blue.", "look it up")
    assert [(c["type"], c["source"], c["text"]) for c in step["citations"]] == [
        ("url_citation", web, "Sky"),
        ("file_citation", "file-1", "notes.txt"),
    ]
