"""Agent configuration that has to reach the model request.

Whether a request is well formed is for the provider to judge, and the live lane
lets the real providers judge it. These checks pin only the settings a user
configures and the agent must carry onto the request: hosted tools, tool search,
completion options for training, citations, Claude's image handling and the
computer-use tool each model gets.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import zlib
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
import pytest
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from PIL import Image
from pydantic import ValidationError

from hud import Environment
from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent, OpenAIChatAgent
from hud.agents.claude import ClaudeToolSearchTool, ClaudeWebFetchTool, ClaudeWebSearchTool
from hud.agents.gemini.tools import (
    GeminiCodeExecutionTool,
    GeminiGoogleSearchTool,
    GeminiUrlContextTool,
)
from hud.agents.openai.tools import OpenAICodeInterpreterTool, OpenAIToolSearchTool
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIChatConfig, OpenAIConfig
from hud.capabilities import Capability
from hud.eval import LocalRuntime, Task
from tests.agents.support import image, mcp_server, run_task, workspace_env
from tests.fixtures.envs import source
from tests.harness import Turn, call, fake_screen, say

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.harness import HudEnv, ModelRequest, Models


WEB_SEARCH = ClaudeWebSearchTool(max_uses=2, allowed_domains=["example.com"])
WEB_FETCH = ClaudeWebFetchTool(max_content_tokens=100, citations_enabled=True)
GROUNDING = [GeminiGoogleSearchTool(), GeminiUrlContextTool(), GeminiCodeExecutionTool()]


@pytest.mark.parametrize(
    ("agent", "sent"),
    [
        pytest.param(
            lambda: ClaudeAgent(
                ClaudeConfig(model="claude-sonnet-4-6", hosted_tools=[WEB_SEARCH, WEB_FETCH])
            ),
            [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 2,
                    "allowed_domains": ["example.com"],
                },
                {
                    "type": "web_fetch_20250910",
                    "name": "web_fetch",
                    "max_content_tokens": 100,
                    "citations": {"enabled": True},
                },
            ],
            id="claude-web-search-and-fetch",
        ),
        pytest.param(
            lambda: OpenAIAgent(
                OpenAIConfig(
                    model="gpt-5.6",
                    hosted_tools=[OpenAICodeInterpreterTool(container={"type": "auto"})],
                )
            ),
            [{"type": "code_interpreter", "container": {"type": "auto"}}],
            id="openai-code-interpreter",
        ),
        pytest.param(
            lambda: GeminiAgent(
                GeminiConfig(model="gemini-3.1-pro-preview", hosted_tools=GROUNDING)
            ),
            [{"googleSearch": {}}, {"urlContext": {}}, {"codeExecution": {}}],
            id="gemini-grounding",
        ),
    ],
)
async def test_configured_hosted_tools_join_the_request(
    agent: Callable[[], Agent],
    sent: list[dict[str, Any]],
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])

    await run_task(workspace_env(tmp_path / "ws"), agent())

    (request,) = models.requests()
    assert [tool for tool in sent if tool not in request.body["tools"]] == []


async def test_a_hosted_tool_the_model_does_not_support_stays_out(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])
    opus_only = ClaudeWebSearchTool(supported_models=("claude-opus-*",))
    agent = ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6", hosted_tools=[opus_only]))

    await run_task(workspace_env(tmp_path / "ws"), agent)

    (request,) = models.requests()
    assert "web_search" not in request.tools


@pytest.mark.parametrize(
    "agent",
    [
        lambda threshold: ClaudeAgent(
            ClaudeConfig(
                model="claude-sonnet-4-6",
                hosted_tools=[ClaudeToolSearchTool(threshold=threshold)],
            )
        ),
        lambda threshold: OpenAIAgent(
            OpenAIConfig(model="gpt-5.6", hosted_tools=[OpenAIToolSearchTool(threshold=threshold)])
        ),
    ],
    ids=["claude", "openai"],
)
@pytest.mark.parametrize("threshold", [1, 5], ids=["over-threshold", "under-threshold"])
async def test_tool_search_defers_mcp_tools_once_they_outnumber_its_threshold(
    agent: Callable[[int], Agent],
    threshold: int,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])

    await Task(env="servers", id="look_up_both").run(
        agent(threshold), runtime=LocalRuntime(source("servers"))
    )

    (request,) = models.requests()
    later = {tool["name"] for tool in request.body["tools"] if tool.get("defer_loading")}
    assert later == ({"database__lookup", "search__lookup"} if threshold == 1 else set())


INVALID_HOSTED = {
    "claude-both-domain-lists": (
        ClaudeAgent,
        ClaudeConfig,
        ClaudeWebSearchTool(allowed_domains=["a.com"], blocked_domains=["b.com"]),
        "Use either allowed_domains or blocked_domains, not both.",
    ),
    "claude-fetch-both-domain-lists": (
        ClaudeAgent,
        ClaudeConfig,
        ClaudeWebFetchTool(allowed_domains=["a.com"], blocked_domains=["b.com"]),
        "Use either allowed_domains or blocked_domains, not both.",
    ),
    "gemini-dynamic-threshold": (
        GeminiAgent,
        GeminiConfig,
        GeminiGoogleSearchTool(dynamic_threshold=0.5),
        "dynamic_threshold is not supported by Gemini Google Search.",
    ),
}


@pytest.mark.parametrize(
    ("agent_cls", "config_cls", "hosted", "error"),
    INVALID_HOSTED.values(),
    ids=INVALID_HOSTED.keys(),
)
async def test_an_invalid_hosted_tool_fails_the_run_before_any_request(
    agent_cls: Any,
    config_cls: Any,
    hosted: Any,
    error: str,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")

    run = await run_task(
        workspace_env(tmp_path / "ws"), agent_cls(config_cls(hosted_tools=[hosted]))
    )

    assert (run.trace.status, run.trace.error) == ("error", f"[agent loop] ValueError: {error}")
    assert models.requests() == []


def request_options(request: ModelRequest) -> dict[str, Any]:
    """A Chat Completions request without its conversation and tool list."""
    return {key: value for key, value in request.body.items() if key not in {"messages", "tools"}}


BASE = {"model": "qwen3.6-plus", "stream": False}
TRAINING = {
    "token-ids-continue-the-kv-cache": (
        OpenAIChatConfig(
            model="qwen3.6-plus",
            completion_kwargs={"extra_body": {"return_token_ids": True}, "temperature": 0.2},
        ),
        [
            Turn(
                tool_calls=call("bash", command="true").tool_calls,
                extra={"prompt_token_ids": [1, 2], "token_ids": [3]},
            ),
            Turn(text="done", extra={"prompt_token_ids": [1, 2, 3, 4], "token_ids": [5]}),
        ],
        [
            BASE | {"logprobs": True, "temperature": 0.2, "return_token_ids": True},
            BASE
            | {
                "logprobs": True,
                "temperature": 0.2,
                "return_token_ids": True,
                "prompt_token_ids": [1, 2, 3],
                "continuation_from": 2,
            },
        ],
    ),
    "multimodal-prompt-chunks-reset-the-continuation": (
        OpenAIChatConfig(
            model="qwen3.6-plus", completion_kwargs={"extra_body": {"return_token_ids": True}}
        ),
        [
            Turn(
                tool_calls=call("bash", command="true").tool_calls,
                extra={"prompt_chunks": [{"type": "text", "text": "hi"}], "token_ids": [3]},
            ),
            say("done"),
        ],
        [BASE | {"logprobs": True, "return_token_ids": True}] * 2,
    ),
    "checkpoint-routes-in-the-body": (
        OpenAIChatConfig(model="qwen3.6-plus", checkpoint="ckpt-7"),
        [say("done")],
        [BASE | {"checkpoint": "ckpt-7"}],
    ),
    "reserved-completion-kwargs-are-dropped": (
        OpenAIChatConfig(
            model="qwen3.6-plus",
            completion_kwargs={
                "model": "other",
                "stream": True,
                "tools": [],
                "messages": [],
                "max_tokens": 50,
            },
        ),
        [say("done")],
        [BASE | {"max_tokens": 50}],
    ),
}


@pytest.mark.parametrize(("config", "turns", "expected"), TRAINING.values(), ids=TRAINING.keys())
async def test_chat_completion_options_reach_every_request(
    config: OpenAIChatConfig,
    turns: list[Turn],
    expected: list[dict[str, Any]],
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script(turns)

    await run_task(workspace_env(tmp_path / "ws"), OpenAIChatAgent(config))

    assert [request_options(request) for request in models.requests()] == expected


CITATIONS = {
    "openai": (
        lambda: OpenAIAgent(OpenAIConfig(model="gpt-5.6", citations_enabled=True)),
        lambda body: body.get("include"),
        ["web_search_call.action.sources"],
    ),
    "gemini": (
        lambda: GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview", citations_enabled=True)),
        lambda body: [tool for tool in body["tools"] if "functionDeclarations" not in tool],
        [{"googleSearch": {}}],
    ),
}


@pytest.mark.parametrize(("agent", "part", "expected"), CITATIONS.values(), ids=CITATIONS.keys())
async def test_citations_ask_the_provider_for_sources(
    agent: Callable[[], Agent],
    part: Callable[[dict[str, Any]], Any],
    expected: Any,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])

    await run_task(workspace_env(tmp_path / "ws"), agent())

    (request,) = models.requests()
    assert part(request.body) == expected


@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param(
            {"max_tool_result_images": 1},
            "requires explicit adaptive or enabled thinking",
            id="image-history-without-thinking",
        ),
        pytest.param(
            {"max_tool_result_images": 1, "thinking": {"type": "disabled"}},
            "requires explicit adaptive or enabled thinking",
            id="image-history-with-thinking-disabled",
        ),
        pytest.param(
            {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 2048}},
            "budget_tokens must be less than max_tokens",
            id="a-budget-equal-to-max-tokens",
        ),
        pytest.param(
            {"max_tokens": 2048, "thinking": {"type": "enabled", "budget_tokens": 2047}},
            None,
            id="a-budget-below-max-tokens",
        ),
    ],
)
def test_claude_refuses_thinking_settings_it_cannot_send(
    config: dict[str, Any], error: str | None
) -> None:
    if error is None:
        assert ClaudeConfig.model_validate(config).thinking == config["thinking"]
        return
    with pytest.raises(ValidationError, match=error):
        ClaudeConfig.model_validate(config)


def png_claiming(size: tuple[int, int]) -> bytes:
    """A 1x1 PNG whose header claims ``size``."""
    data = bytearray(image((1, 1)))
    data[16:24] = struct.pack(">II", *size)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    return bytes(data)


def rotated_jpeg() -> bytes:
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise for display
    return image((40, 20), format="JPEG", color="red", exif=exif)


# Claude reads images up to 1568 pixels on a side and 1.15 megapixels; larger ones
# are scaled down to fit, keeping their shape.
CLAUDE_VIEWS: dict[str, tuple[bytes, tuple[int, int] | str]] = {
    "small-png": (image((40, 30)), (40, 30)),
    "wide-png": (image((3000, 2000)), (3000, 2000)),
    "tall-png": (image((2000, 4000)), (2000, 4000)),
    "exif-jpeg-is-turned-upright": (rotated_jpeg(), (20, 40)),
    "gif": (image((32, 24), format="GIF"), (32, 24)),
    "oversized-header": (png_claiming((4001, 4000)), "exceeds the 16,000,000-pixel source limit"),
    "corrupt-png": (b"\x89PNG\r\n\x1a\n" + b"garbage", "Cannot view image"),
    "binary": (b"\x00binary", "expected UTF-8 text or an image"),
}


@pytest.mark.filterwarnings("ignore:Image size.*:PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize(("data", "expected"), CLAUDE_VIEWS.values(), ids=CLAUDE_VIEWS.keys())
async def test_an_image_claude_views_is_sent_within_its_limits(
    data: bytes,
    expected: tuple[int, int] | str,
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    root = tmp_path / "ws"
    models.script(
        [call("str_replace_based_edit_tool", command="view", path=f"{root}/file"), say("done")]
    )

    await run_task(
        workspace_env(root, files={"file": data}),
        ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6")),
    )

    (result,) = models.requests()[-1].body["messages"][-1]["content"]
    (block,) = result["content"]
    if isinstance(expected, str):
        assert (result["is_error"], expected in block["text"]) == (True, True)
        return
    with Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))) as sent:
        width, height = sent.size
    assert max(width, height) <= 1568 and width * height <= 1_150_000
    assert abs(width / height - expected[0] / expected[1]) < 0.01


def screenshots() -> FastMCP:
    server = FastMCP("screenshots")

    @server.tool
    def screenshot() -> ToolResult:
        """Capture the current screen."""
        data = base64.b64encode(image((32, 24))).decode()
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text="screen"),
                mcp_types.ImageContent(type="image", mimeType="image/png", data=data),
            ]
        )

    return server


@pytest.mark.parametrize("limit", [None, 1], ids=["unlimited", "newest-image-only"])
async def test_claude_sends_only_its_newest_tool_result_images_when_asked(
    limit: int | None, models: Models, hud_env: HudEnv
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script(
        [
            replace(call("screenshot"), reasoning="reason-1"),
            replace(call("screenshot"), reasoning="reason-2"),
            say("done"),
        ]
    )
    agent = ClaudeAgent(
        ClaudeConfig(
            model="claude-sonnet-4-6",
            max_tool_result_images=limit,
            thinking={"type": "adaptive"} if limit is not None else None,
        )
    )

    async with mcp_server(screenshots(), name="screenshots") as tools:
        env = Environment("shots", capabilities=[tools])

        @env.template()
        async def task():
            yield "Take two screenshots."
            yield 1.0

        run = await run_task(env, agent)

    last = models.requests()[-1]
    sent = [
        [block["type"] for block in message["content"][0]["content"]]
        for message in last.body["messages"][2::2]
    ]
    assert run.trace.status == "completed"
    if limit is None:
        assert sent == [["text", "image"], ["text", "image"]]
        assert "anthropic-beta" not in last.headers
    else:
        assert sent == [["text", "text"], ["text", "image"]]
        assert "thinking-binding-controls-2026-08-01" in last.headers["anthropic-beta"]


def long_output() -> FastMCP:
    server = FastMCP("long")

    @server.tool()
    def long_output() -> str:
        """Return a long text."""
        return "h" * 500 + "t" * 500

    return server


async def test_gemini_bounds_a_long_tool_output_to_its_head_and_tail(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([call("long_output"), say("done")])
    agent = GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview", max_tool_result_chars=100))

    async with mcp_server(long_output()) as tools:
        await run_task(workspace_env(tmp_path / "ws", capabilities=(tools,)), agent)

    (result,) = models.requests()[-1].tool_results
    output = json.loads(result)["output"]
    assert output.startswith("Output too large: showing the first 20 and last 80 characters")
    assert output.endswith("h" * 20 + "\n\n... [900 characters omitted] ...\n\n" + "t" * 80)


# Claude models get the newest computer-use tool they support with its beta; other
# models served over the Messages API get a plain schema describing the display.
CLAUDE_COMPUTERS = {
    "claude-sonnet-4-6": ("computer-use-2025-11-24", "computer_20251124"),
    "claude-sonnet-4-5": ("computer-use-2025-01-24", "computer_20250124"),
    "claude-future-model": ("computer-use-2025-11-24", "computer_20251124"),
    "qwen/qwen3.8-max": (None, None),
}


@pytest.mark.parametrize(
    ("model", "advertised"), CLAUDE_COMPUTERS.items(), ids=CLAUDE_COMPUTERS.keys()
)
async def test_claude_advertises_the_computer_tool_its_model_supports(
    model: str,
    advertised: tuple[str | None, str | None],
    models: Models,
    hud_env: HudEnv,
    tmp_path: Path,
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])

    async with fake_screen(width=200, height=100) as screen:
        screen_capability = Capability.rfb(url=screen.url)
        await run_task(
            workspace_env(tmp_path / "ws", capabilities=(screen_capability,)),
            ClaudeAgent(ClaudeConfig(model=model)),
        )

    (request,) = models.requests()
    computer = next(tool for tool in request.body["tools"] if tool["name"] == "computer")
    assert (request.headers.get("anthropic-beta"), computer.get("type")) == advertised
    if advertised[1] is not None:
        assert (computer["display_width_px"], computer["display_height_px"]) == (200, 100)
    else:
        assert "display is 200x100 pixels" in computer["description"]


async def test_gemini_withholds_the_predefined_functions_it_is_configured_to_exclude(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])
    agent = GeminiAgent(
        GeminiConfig(
            model="gemini-3.1-pro-preview",
            excluded_predefined_functions=["drag_and_drop", "search"],
        )
    )

    async with fake_screen() as screen:
        screen_capability = Capability.rfb(url=screen.url)
        await run_task(workspace_env(tmp_path / "ws", capabilities=(screen_capability,)), agent)

    (request,) = models.requests()
    (computer,) = [tool["computerUse"] for tool in request.body["tools"] if "computerUse" in tool]
    assert computer["excluded_predefined_functions"] == ["drag_and_drop", "search"]


@pytest.mark.parametrize(("configured", "sent"), [({}, None), ({"max_output_tokens": 4096}, 4096)])
async def test_gemini_leaves_the_output_cap_to_the_model_unless_one_is_configured(
    configured: dict[str, Any], sent: int | None, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])
    agent = GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview", **configured))

    await run_task(workspace_env(tmp_path / "ws"), agent)

    (request,) = models.requests()
    assert request.body.get("generationConfig", {}).get("maxOutputTokens") == sent


def named(name: str, *tools: str) -> FastMCP:
    mcp = FastMCP(name)
    for tool in tools:
        mcp.tool(name=tool, description=f"Run {tool}.")(lambda: tool)
    return mcp


async def test_mcp_tools_are_qualified_within_64_characters_and_a_clash_fails_the_run(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])
    long = "lookup_" * 10

    async with (
        mcp_server(named("company.tools", long + "a"), name="company.tools") as dotted,
        mcp_server(named("company-tools", long + "b"), name="company-tools") as dashed,
        mcp_server(named("a", "b__c"), name="a") as first,
        mcp_server(named("a__b", "c"), name="a__b") as second,
    ):
        await run_task(
            workspace_env(tmp_path / "long", capabilities=(dotted, dashed)),
            OpenAIAgent(OpenAIConfig(model="gpt-5.6")),
        )
        clash = workspace_env(tmp_path / "clash", capabilities=(first, second))
        clashed = await run_task(clash, OpenAIAgent(OpenAIConfig(model="gpt-5.6")))

    (request,) = models.requests()
    qualified = [name for name in request.tools if long[:20] in name]
    assert len(qualified) == 2 and all(len(name) <= 64 for name in qualified)
    assert clashed.trace.error == (
        "[agent loop] ValueError: MCP tool name collision after qualification: 'a__b__c'"
    )
