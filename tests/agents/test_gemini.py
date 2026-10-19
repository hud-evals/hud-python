"""What only ``GeminiAgent`` sends and records: its grounding tools and output cap in the
generateContent request, the computer functions it leaves out, how it bounds a long tool
output it sends back, and the grounding it reads back as citations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import FastMCP

from hud.agents import GeminiAgent
from hud.agents.gemini.tools import (
    GeminiCodeExecutionTool,
    GeminiGoogleSearchTool,
    GeminiUrlContextTool,
)
from hud.agents.types import GeminiConfig
from hud.capabilities import Capability
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Turn, call, fake_screen, mcp_server, say, steps

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import HudEnv, ModelRequest, Models


def gemini(**config: Any) -> GeminiAgent:
    return GeminiAgent(GeminiConfig.model_validate({"model": "gemini-3.1-pro-preview", **config}))


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


async def first_request(
    models: Models, root: Path, capabilities: tuple[Capability, ...] = (), **config: Any
) -> ModelRequest:
    models.script([say("done")])
    await run_task(workspace_env(root, capabilities=capabilities), gemini(**config))
    return models.requests()[0]


def hosted(request: ModelRequest) -> list[dict[str, Any]]:
    return [tool for tool in request.body["tools"] if "functionDeclarations" not in tool]


@pytest.mark.parametrize(
    ("config", "tools"),
    [
        pytest.param(
            {
                "hosted_tools": [
                    GeminiGoogleSearchTool(),
                    GeminiUrlContextTool(),
                    GeminiCodeExecutionTool(),
                ]
            },
            [{"googleSearch": {}}, {"urlContext": {}}, {"codeExecution": {}}],
            id="hosted-grounding-tools",
        ),
        pytest.param(
            {"citations_enabled": True},
            [{"googleSearch": {}}],
            id="citations-ask-for-google-search",
        ),
    ],
)
async def test_the_hosted_tools_its_configuration_names_are_offered(
    config: dict[str, Any], tools: list[dict[str, Any]], models: Models, tmp_path: Path
) -> None:
    request = await first_request(models, tmp_path / "ws", **config)

    assert hosted(request) == tools


async def test_excluded_computer_functions_are_left_out_of_the_computer_tool(
    models: Models, tmp_path: Path
) -> None:
    async with fake_screen() as screen:
        request = await first_request(
            models,
            tmp_path / "ws",
            (Capability.rfb(url=screen.url),),
            excluded_predefined_functions=["drag_and_drop", "search"],
        )

    (computer,) = hosted(request)
    assert computer["computerUse"]["excluded_predefined_functions"] == ["drag_and_drop", "search"]


@pytest.mark.parametrize(("cap", "sent"), [(None, None), (4096, 4096)], ids=["unset", "configured"])
async def test_the_output_cap_is_sent_only_when_configured(
    cap: int | None, sent: int | None, models: Models, tmp_path: Path
) -> None:
    config: dict[str, Any] = {} if cap is None else {"max_output_tokens": cap}

    request = await first_request(models, tmp_path / "ws", **config)

    assert request.body.get("generationConfig", {}).get("maxOutputTokens") == sent


async def test_a_long_tool_output_is_sent_back_with_its_head_and_tail(
    models: Models, tmp_path: Path
) -> None:
    server = FastMCP("long")
    server.tool(name="long_output", description="Return a long text.")(
        lambda: "h" * 500 + "t" * 500
    )
    models.script([call("long_output"), say("done")])

    async with mcp_server(server) as long_output:
        env = workspace_env(tmp_path / "ws", capabilities=(long_output,))
        await run_task(env, gemini(max_tool_result_chars=100))

    (result,) = models.requests()[-1].tool_results
    output = json.loads(result)["output"]
    assert output.startswith("Output too large: showing the first 20 and last 80 characters")
    assert output.endswith("h" * 20 + "\n\n... [900 characters omitted] ...\n\n" + "t" * 80)


async def test_grounding_lands_on_the_agent_step_as_citations(
    models: Models, tmp_path: Path
) -> None:
    web = "https://example.com/sky"
    models.script(
        [
            Turn(
                text="Blue.",
                reasoning="look it up",
                extra={
                    "groundingMetadata": {
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
                },
            )
        ]
    )

    run = await run_task(workspace_env(tmp_path / "ws"), gemini())

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert (step["content"], step["reasoning"]) == ("Blue.", "look it up")
    assert [(c["type"], c["source"], c["text"]) for c in step["citations"]] == [
        ("grounding", web, "Blue."),
        ("grounding", "https://example.com/unused", ""),
    ]
