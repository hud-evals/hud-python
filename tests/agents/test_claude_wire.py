"""What only ``ClaudeAgent`` puts on the wire: its configuration in the Messages request,
the computer tool each model gets, the images it sends back, and Bedrock invocation.

Whether a request is well formed is for the provider to judge, and the live lane
lets the real one judge it. These rows pin what a user configures and the agent
must carry onto the request.
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
import pytest
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from PIL import Image

from hud.agents import ClaudeAgent
from hud.agents.claude import ClaudeToolSearchTool, ClaudeWebFetchTool, ClaudeWebSearchTool
from hud.agents.types import ClaudeConfig
from hud.capabilities import Capability
from tests.agents.families import CLAUDE, Wire, carried, two_servers
from tests.fixtures.environments import run_task, workspace_env
from tests.support import call, fake_screen, mcp_server, say

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from tests.support import FakeServices, HudEnv, ModelRequest, Models


def image(
    size: tuple[int, int],
    *,
    format: str = "PNG",
    mode: str = "RGB",
    color: tuple[int, ...] | str = "white",
    exif: Image.Exif | None = None,
) -> bytes:
    """An image of ``size`` encoded as ``format``."""
    buffer = io.BytesIO()
    options: dict[str, Any] = {} if exif is None else {"exif": exif}
    Image.new(mode, size, color).save(buffer, format=format, **options)
    return buffer.getvalue()


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


@asynccontextmanager
async def screenshot_server() -> AsyncIterator[tuple[Capability, ...]]:
    async with mcp_server(screenshots(), name="screenshots") as server:
        yield (server,)


@asynccontextmanager
async def screen() -> AsyncIterator[tuple[Capability, ...]]:
    async with fake_screen(width=200, height=100) as display:
        yield (Capability.rfb(url=display.url),)


def tools(requests: list[ModelRequest]) -> list[dict[str, Any]]:
    (request,) = requests
    return request.body["tools"]


def deferred(requests: list[ModelRequest]) -> set[str]:
    return {tool["name"] for tool in tools(requests) if tool.get("defer_loading")}


def tool_result_images(requests: list[ModelRequest]) -> tuple[list[list[str]], bool]:
    """The block types of each tool result the last request sends back, and whether it
    asks for the binding-controls beta."""
    last = requests[-1]
    sent = [
        [block["type"] for block in message["content"][0]["content"]]
        for message in last.body["messages"][2::2]
    ]
    return sent, "thinking-binding-controls-2026-08-01" in last.headers.get("anthropic-beta", "")


def computer(requests: list[ModelRequest]) -> tuple[str | None, str | None, str]:
    """The beta header, the computer tool's type, and how it describes the 200x100 display."""
    (request,) = requests
    tool = next(tool for tool in request.body["tools"] if tool["name"] == "computer")
    display = (
        f"{tool['display_width_px']}x{tool['display_height_px']}"
        if "display_width_px" in tool
        else ("described" if "display is 200x100 pixels" in tool["description"] else "missing")
    )
    return request.headers.get("anthropic-beta"), tool.get("type"), display


TWO_SCREENSHOTS = [
    replace(call("screenshot"), reasoning="reason-1"),
    replace(call("screenshot"), reasoning="reason-2"),
    say("done"),
]

WIRE = {
    "hosted-web-search-and-fetch": Wire(
        {"hosted_tools": [ClaudeWebSearchTool(), ClaudeWebFetchTool(citations_enabled=True)]},
        lambda requests: [t for t in tools(requests) if t.get("type", "").startswith("web_")],
        [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "web_fetch_20250910", "name": "web_fetch", "citations": {"enabled": True}},
        ],
    ),
    "a-hosted-tool-for-other-models-stays-out": Wire(
        {"hosted_tools": [ClaudeWebSearchTool(supported_models=("claude-opus-*",))]},
        lambda requests: "web_search" in requests[0].tools,
        False,
    ),
    "tool-search-defers-mcp-tools-over-its-threshold": Wire(
        {"hosted_tools": [ClaudeToolSearchTool(threshold=1)]},
        deferred,
        {"database__lookup", "search__lookup"},
        serve=two_servers,
    ),
    "tool-search-under-its-threshold-defers-nothing": Wire(
        {"hosted_tools": [ClaudeToolSearchTool(threshold=5)]}, deferred, set(), serve=two_servers
    ),
    "every-tool-result-image-by-default": Wire(
        {},
        tool_result_images,
        ([["text", "image"], ["text", "image"]], False),
        turns=TWO_SCREENSHOTS,
        serve=screenshot_server,
    ),
    "only-the-newest-tool-result-image": Wire(
        {"max_tool_result_images": 1, "thinking": {"type": "adaptive"}},
        tool_result_images,
        ([["text", "text"], ["text", "image"]], True),
        turns=TWO_SCREENSHOTS,
        serve=screenshot_server,
    ),
    "computer-tool-for-claude-sonnet-4-6": Wire(
        {"model": "claude-sonnet-4-6"},
        computer,
        ("computer-use-2025-11-24", "computer_20251124", "200x100"),
        serve=screen,
    ),
    "computer-tool-for-claude-sonnet-4-5": Wire(
        {"model": "claude-sonnet-4-5"},
        computer,
        ("computer-use-2025-01-24", "computer_20250124", "200x100"),
        serve=screen,
    ),
    "a-plain-computer-schema-for-other-models": Wire(
        {"model": "qwen/qwen3.8-max"}, computer, (None, None, "described"), serve=screen
    ),
}


@pytest.mark.parametrize("wire", WIRE.values(), ids=WIRE.keys())
async def test_claude_carries_its_configuration_onto_the_request(
    wire: Wire, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")

    assert await carried(CLAUDE, wire, models, tmp_path / "ws") == wire.expected


def rotated_jpeg() -> bytes:
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise for display
    return image((40, 20), format="JPEG", color="red", exif=exif)


# Claude reads images up to 1568 pixels on a side and 1.15 megapixels; larger ones
# are scaled down to fit, keeping their shape.
CLAUDE_VIEWS: dict[str, tuple[bytes, tuple[int, int] | str]] = {
    "small-png": (image((40, 30)), (40, 30)),
    "wide-png": (image((3000, 2000)), (3000, 2000)),
    "exif-jpeg-is-turned-upright": (rotated_jpeg(), (20, 40)),
    "corrupt-png": (b"\x89PNG\r\n\x1a\n" + b"garbage", "Cannot view image"),
}


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

    await run_task(workspace_env(root, files={"file": data}), ClaudeAgent(CLAUDE.config()))

    (result,) = models.requests()[-1].body["messages"][-1]["content"]
    (block,) = result["content"]
    if isinstance(expected, str):
        assert (result["is_error"], expected in block["text"]) == (True, True)
        return
    with Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))) as sent:
        width, height = sent.size
    assert max(width, height) <= 1568 and width * height <= 1_150_000
    assert abs(width / height - expected[0] / expected[1]) < 0.01


async def test_a_bedrock_inference_profile_is_invoked_on_bedrock_without_streaming(
    services: FakeServices, hud_env: HudEnv, tmp_path: Path
) -> None:
    pytest.importorskip("botocore", reason="needs the bedrock extra")
    profile = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/claude"
    hud_env.set(
        AWS_ACCESS_KEY_ID="AKIA",
        AWS_SECRET_ACCESS_KEY="secret",
        AWS_REGION="us-east-1",
        ANTHROPIC_BEDROCK_BASE_URL=services.url("gateway"),
    )
    services.route(
        "gateway",
        "POST",
        "/model/{model:path}",
        json={
            "id": "msg_bedrock",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": [{"type": "text", "text": "from bedrock"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )

    run = await run_task(workspace_env(tmp_path / "ws"), ClaudeAgent(ClaudeConfig(model=profile)))

    (request,) = services.requests("gateway", "POST")
    assert run.trace.content == "from bedrock"
    assert (request.path, request.headers["authorization"].split(" ")[0]) == (
        f"/model/{profile}/invoke",
        "AWS4-HMAC-SHA256",
    )
    body = request.json
    assert (body["anthropic_version"], "model" in body, "stream" in body) == (
        "bedrock-2023-05-31",
        False,
        False,
    )
