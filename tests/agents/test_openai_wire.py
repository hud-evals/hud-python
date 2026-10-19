"""What only ``OpenAIAgent`` puts on the wire: its configuration in the Responses request,
and how it keeps its response chain when it discards a turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from hud.agents.openai.tools import OpenAICodeInterpreterTool, OpenAIToolSearchTool
from tests.agents.families import OPENAI, Wire, carried, two_servers
from tests.support import run_task, say, shell_call, workspace_env

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, ModelRequest, Models


def tools(requests: list[ModelRequest]) -> list[dict[str, Any]]:
    (request,) = requests
    return request.body["tools"]


def deferred(requests: list[ModelRequest]) -> set[str]:
    return {tool["name"] for tool in tools(requests) if tool.get("defer_loading")}


WIRE = {
    "hosted-code-interpreter": Wire(
        {"hosted_tools": [OpenAICodeInterpreterTool(container={"type": "auto"})]},
        lambda requests: [t for t in tools(requests) if t["type"] == "code_interpreter"],
        [{"type": "code_interpreter", "container": {"type": "auto"}}],
    ),
    "tool-search-defers-mcp-tools-over-its-threshold": Wire(
        {"hosted_tools": [OpenAIToolSearchTool(threshold=1)]},
        deferred,
        {"database__lookup", "search__lookup"},
        serve=two_servers,
    ),
    "tool-search-under-its-threshold-defers-nothing": Wire(
        {"hosted_tools": [OpenAIToolSearchTool(threshold=5)]}, deferred, set(), serve=two_servers
    ),
    "citations-ask-for-web-search-sources": Wire(
        {"citations_enabled": True},
        lambda requests: requests[0].body.get("include"),
        ["web_search_call.action.sources"],
    ),
}


@pytest.mark.parametrize("wire", WIRE.values(), ids=WIRE.keys())
async def test_openai_carries_its_configuration_onto_the_request(
    wire: Wire, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")

    assert await carried(OPENAI, wire, models, tmp_path / "ws") == wire.expected


async def test_responses_resamples_an_empty_shell_call_without_advancing_its_chain(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.queue([shell_call("true"), shell_call(), say("done")])

    run = await run_task(workspace_env(tmp_path / "ws"), OPENAI.agent_cls(OPENAI.config()))

    chain = [request.body.get("previous_response_id") for request in models.requests()]
    assert (run.trace.content, chain) == ("done", [None, "resp_1", "resp_1"])
