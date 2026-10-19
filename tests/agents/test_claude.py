"""What only ``ClaudeAgent`` sends and records: its hosted tools and tool search, the
computer tool each model gets, the images it sends back, Bedrock invocation, and the
citations and refusals it reads back."""

from __future__ import annotations

import base64
import io
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
from tests.fixtures.environments import run_task, workspace_env
from tests.support import Turn, call, fake_screen, mcp_server, say, steps

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import FakeServices, HudEnv, ModelRequest, Models


def claude(**config: Any) -> ClaudeAgent:
    return ClaudeAgent(ClaudeConfig.model_validate({"model": "claude-sonnet-4-6", **config}))


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


async def first_request(
    models: Models, root: Path, capabilities: tuple[Capability, ...] = (), **config: Any
) -> ModelRequest:
    models.script([say("done")])
    await run_task(workspace_env(root, capabilities=capabilities), claude(**config))
    return models.requests()[0]


def image(
    size: tuple[int, int],
    *,
    format: str = "PNG",
    color: str = "white",
    exif: Image.Exif | None = None,
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(
        buffer, format=format, **({} if exif is None else {"exif": exif})
    )
    return buffer.getvalue()


# ─── the tools it offers ───────────────────────────────────────────────


async def test_hosted_web_search_and_fetch_are_offered_as_configured(
    models: Models, tmp_path: Path
) -> None:
    hosted = [ClaudeWebSearchTool(), ClaudeWebFetchTool(citations_enabled=True)]

    request = await first_request(models, tmp_path / "ws", hosted_tools=hosted)

    assert [t for t in request.body["tools"] if t.get("type", "").startswith("web_")] == [
        {"type": "web_search_20250305", "name": "web_search"},
        {"type": "web_fetch_20250910", "name": "web_fetch", "citations": {"enabled": True}},
    ]


async def test_a_hosted_tool_for_other_models_stays_out(models: Models, tmp_path: Path) -> None:
    hosted = [ClaudeWebSearchTool(supported_models=("claude-opus-*",))]

    request = await first_request(models, tmp_path / "ws", hosted_tools=hosted)

    assert "web_search" not in request.tools


def lookups() -> FastMCP:
    server = FastMCP("lookups")
    server.tool(name="lookup", description="Look a key up.")(lambda key: key)
    return server


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
            hosted_tools=[ClaudeToolSearchTool(threshold=threshold)],
        )

    assert {tool["name"] for tool in request.body["tools"] if tool.get("defer_loading")} == deferred


@pytest.mark.parametrize(
    ("model", "beta", "kind", "display"),
    [
        pytest.param(
            "claude-sonnet-4-6", "computer-use-2025-11-24", "computer_20251124", "200x100", id="4-6"
        ),
        pytest.param(
            "claude-sonnet-4-5", "computer-use-2025-01-24", "computer_20250124", "200x100", id="4-5"
        ),
        pytest.param("qwen/qwen3.8-max", None, None, "described", id="another-model"),
    ],
)
async def test_each_model_gets_its_own_computer_tool_for_the_screen(
    model: str,
    beta: str | None,
    kind: str | None,
    display: str,
    models: Models,
    tmp_path: Path,
) -> None:
    async with fake_screen(width=200, height=100) as screen:
        request = await first_request(
            models, tmp_path / "ws", (Capability.rfb(url=screen.url),), model=model
        )

    tool = next(tool for tool in request.body["tools"] if tool["name"] == "computer")
    shown = (
        f"{tool['display_width_px']}x{tool['display_height_px']}"
        if "display_width_px" in tool
        else ("described" if "display is 200x100 pixels" in tool["description"] else "missing")
    )
    assert (request.headers.get("anthropic-beta"), tool.get("type"), shown) == (beta, kind, display)


async def test_a_tool_result_that_failed_is_marked_as_an_error_for_the_model(
    models: Models, tmp_path: Path
) -> None:
    models.script([call("str_replace_based_edit_tool", command="bogus", path="/x"), say("done")])

    await run_task(workspace_env(tmp_path / "ws"), claude())

    (result,) = models.requests()[-1].body["messages"][-1]["content"]
    assert (result["type"], result["is_error"]) == ("tool_result", True)


# ─── the images it sends back ──────────────────────────────────────────


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


@pytest.mark.parametrize(
    ("config", "sent", "beta"),
    [
        pytest.param(
            {}, [["text", "image"], ["text", "image"]], False, id="every-image-by-default"
        ),
        pytest.param(
            {"max_tool_result_images": 1, "thinking": {"type": "adaptive"}},
            [["text", "text"], ["text", "image"]],
            True,
            id="only-the-newest-image",
        ),
    ],
)
async def test_tool_result_images_are_sent_back_up_to_the_configured_count(
    config: dict[str, Any], sent: list[list[str]], beta: bool, models: Models, tmp_path: Path
) -> None:
    models.script(
        [
            replace(call("screenshot"), reasoning="reason-1"),
            replace(call("screenshot"), reasoning="reason-2"),
            say("done"),
        ]
    )

    async with mcp_server(screenshots(), name="screenshots") as server:
        env = workspace_env(tmp_path / "ws", capabilities=(server,))
        await run_task(env, claude(**config))

    last = models.requests()[-1]
    blocks = [
        [block["type"] for block in message["content"][0]["content"]]
        for message in last.body["messages"][2::2]
    ]
    assert blocks == sent
    assert (
        "thinking-binding-controls-2026-08-01" in last.headers.get("anthropic-beta", "")
    ) is beta


def rotated_jpeg() -> bytes:
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise for display
    return image((40, 20), format="JPEG", color="red", exif=exif)


# Claude reads images up to 1568 pixels on a side and 1.15 megapixels; larger ones are
# scaled down to fit, keeping their shape.
@pytest.mark.parametrize(
    ("data", "shape"),
    [
        pytest.param(image((40, 30)), (40, 30), id="small-png"),
        pytest.param(image((3000, 2000)), (3000, 2000), id="wide-png"),
        pytest.param(rotated_jpeg(), (20, 40), id="an-exif-jpeg-is-turned-upright"),
        pytest.param(b"\x89PNG\r\n\x1a\n" + b"garbage", None, id="a-corrupt-png"),
    ],
)
async def test_an_image_the_editor_views_is_sent_within_claudes_limits(
    data: bytes, shape: tuple[int, int] | None, models: Models, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    models.script(
        [call("str_replace_based_edit_tool", command="view", path=f"{root}/file"), say("done")]
    )

    await run_task(workspace_env(root, files={"file": data}), claude())

    (result,) = models.requests()[-1].body["messages"][-1]["content"]
    (block,) = result["content"]
    if shape is None:
        assert (result["is_error"], "Cannot view image" in block["text"]) == (True, True)
        return
    with Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))) as sent:
        width, height = sent.size
    assert max(width, height) <= 1568 and width * height <= 1_150_000
    assert abs(width / height - shape[0] / shape[1]) < 0.01


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

    run = await run_task(workspace_env(tmp_path / "ws"), claude(model=profile))

    (request,) = services.requests("gateway", "POST")
    assert run.trace.content == "from bedrock"
    assert (request.path, request.headers["authorization"].split(" ")[0]) == (
        f"/model/{profile}/invoke",
        "AWS4-HMAC-SHA256",
    )
    assert (
        request.json["anthropic_version"],
        "model" in request.json,
        "stream" in request.json,
    ) == (
        "bedrock-2023-05-31",
        False,
        False,
    )


# ─── what its replies record ───────────────────────────────────────────


async def test_document_and_web_citations_land_on_the_agent_step(
    models: Models, tmp_path: Path
) -> None:
    web = "https://example.com/sky"
    citations = [
        {
            "type": "char_location",
            "cited_text": "the sky is blue",
            "document_index": 0,
            "document_title": "notes",
            "start_char_index": 4,
            "end_char_index": 19,
        },
        {
            "type": "web_search_result_location",
            "cited_text": "blue skies",
            "url": web,
            "title": "Sky",
            "encrypted_index": "x",
        },
    ]
    models.script(
        [
            Turn(
                reasoning="look it up",
                native=({"type": "text", "text": "Blue.", "citations": citations},),
            )
        ]
    )

    run = await run_task(workspace_env(tmp_path / "ws"), claude())

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert (step["content"], step["reasoning"]) == ("Blue.", "look it up")
    assert [(c["type"], c["source"], c["text"]) for c in step["citations"]] == [
        ("document_citation", "0", "the sky is blue"),
        ("web_search_result_location", web, "blue skies"),
    ]


@pytest.mark.parametrize(
    ("explanation", "refusal"),
    [
        (
            "This request was blocked under the Usage Policy.",
            "This request was blocked under the Usage Policy.",
        ),
        (None, "This request was refused (cyber)."),
    ],
    ids=["with-explanation", "category-only"],
)
async def test_a_refusal_lands_on_the_agent_step(
    explanation: str | None, refusal: str, models: Models, tmp_path: Path
) -> None:
    details = {"type": "refusal", "category": "cyber", "explanation": explanation}
    models.script([Turn(finish="refusal", extra={"stop_details": details})])

    run = await run_task(workspace_env(tmp_path / "ws"), claude())

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert (step["finish_reason"], step["refusal"]) == ("refusal", refusal)
