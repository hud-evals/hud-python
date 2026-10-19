"""What only ``GeminiAgent`` puts on the wire: its configuration in the generateContent
request, and how it bounds a long tool output it sends back."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import FastMCP

from hud.agents.gemini.tools import (
    GeminiCodeExecutionTool,
    GeminiGoogleSearchTool,
    GeminiUrlContextTool,
)
from hud.capabilities import Capability
from tests.agents.families import GEMINI, Wire, carried
from tests.support import call, fake_screen, mcp_server, say

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from tests.support import HudEnv, ModelRequest, Models


def hosted(requests: list[ModelRequest]) -> list[dict[str, Any]]:
    (request,) = requests
    return [tool for tool in request.body["tools"] if "functionDeclarations" not in tool]


@asynccontextmanager
async def screen() -> AsyncIterator[tuple[Capability, ...]]:
    async with fake_screen() as display:
        yield (Capability.rfb(url=display.url),)


def long_output() -> FastMCP:
    server = FastMCP("long")

    @server.tool()
    def long_output() -> str:
        """Return a long text."""
        return "h" * 500 + "t" * 500

    return server


@asynccontextmanager
async def long_output_server() -> AsyncIterator[tuple[Capability, ...]]:
    async with mcp_server(long_output()) as server:
        yield (server,)


HEAD = "Output too large: showing the first 20 and last 80 characters"
TAIL = "h" * 20 + "\n\n... [900 characters omitted] ...\n\n" + "t" * 80


def bounded(requests: list[ModelRequest]) -> tuple[bool, bool]:
    """Whether the tool output sent back opens with the notice and ends with the kept text."""
    (result,) = requests[-1].tool_results
    output = json.loads(result)["output"]
    return output.startswith(HEAD), output.endswith(TAIL)


WIRE = {
    "hosted-grounding-tools": Wire(
        {
            "hosted_tools": [
                GeminiGoogleSearchTool(),
                GeminiUrlContextTool(),
                GeminiCodeExecutionTool(),
            ]
        },
        hosted,
        [{"googleSearch": {}}, {"urlContext": {}}, {"codeExecution": {}}],
    ),
    "citations-ask-for-google-search": Wire(
        {"citations_enabled": True}, hosted, [{"googleSearch": {}}]
    ),
    "excluded-predefined-functions": Wire(
        {"excluded_predefined_functions": ["drag_and_drop", "search"]},
        lambda requests: [
            tool["computerUse"]["excluded_predefined_functions"] for tool in hosted(requests)
        ],
        [["drag_and_drop", "search"]],
        serve=screen,
    ),
    "the-output-cap-is-left-to-the-model": Wire(
        {},
        lambda requests: requests[0].body.get("generationConfig", {}).get("maxOutputTokens"),
        None,
    ),
    "a-configured-output-cap": Wire(
        {"max_output_tokens": 4096},
        lambda requests: requests[0].body.get("generationConfig", {}).get("maxOutputTokens"),
        4096,
    ),
    "a-long-tool-output-keeps-its-head-and-tail": Wire(
        {"max_tool_result_chars": 100},
        bounded,
        (True, True),
        turns=[call("long_output"), say("done")],
        serve=long_output_server,
    ),
}


@pytest.mark.parametrize("wire", WIRE.values(), ids=WIRE.keys())
async def test_gemini_carries_its_configuration_onto_the_request(
    wire: Wire, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")

    assert await carried(GEMINI, wire, models, tmp_path / "ws") == wire.expected
