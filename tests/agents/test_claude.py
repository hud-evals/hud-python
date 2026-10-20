"""What only ``ClaudeAgent`` does: the hosted tools, tool search and computer tool it
offers, how it retries an invalid call or a broken stream, the images it sends back,
Bedrock invocation, and the citations and refusals it records."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import mcp.types as mcp_types
import pytest
from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from PIL import Image

from hud.agents import ClaudeAgent
from hud.agents.claude import ClaudeToolSearchTool, ClaudeWebFetchTool, ClaudeWebSearchTool
from hud.agents.types import ClaudeConfig, ToolStep
from hud.capabilities import Capability
from hud.eval import LocalRuntime, Task, rollout
from tests.conftest import steps
from tests.fixtures.asgi import mcp_server
from tests.fixtures.environments import workspace_env
from tests.fixtures.models import Turn, call, stream_error
from tests.fixtures.screen import fake_screen

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import HudEnv
    from tests.fixtures.models import Models
    from tests.fixtures.services import FakeServices

MODEL = "claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def signed_in(hud_env: HudEnv) -> None:
    hud_env.set(HUD_API_KEY="k")


def png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def upright_after_rotation() -> bytes:
    """A 40x20 JPEG whose EXIF orientation says to show it rotated, as 20x40."""
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "red").save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("model", "offered"),
    [("claude-opus-4-6", ["web_fetch", "web_search"]), (MODEL, ["web_fetch"])],
    ids=["a-model-it-supports", "a-model-it-does-not"],
)
async def test_hosted_web_tools_are_offered_to_the_models_they_support(
    model: str, offered: list[str], models: Models, tmp_path: Path
) -> None:
    hosted = [ClaudeWebFetchTool(), ClaudeWebSearchTool(supported_models=("claude-opus-*",))]
    models.script([Turn(text="done")])

    env = workspace_env(tmp_path / "ws")
    await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=model, hosted_tools=hosted)),
        runtime=LocalRuntime(env),
    )

    (request,) = models.requests()
    web = [
        tool["name"] for tool in request.body["tools"] if tool.get("type", "").startswith("web_")
    ]
    assert web == offered


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
    agent = ClaudeAgent(
        ClaudeConfig(model=MODEL, hosted_tools=[ClaudeToolSearchTool(threshold=threshold)])
    )

    async with (
        mcp_server(lookups, name="database") as database,
        mcp_server(lookups, name="search") as search,
    ):
        env = workspace_env(tmp_path / "ws", capabilities=(database, search))
        await rollout(Task(env=env.name, id="task"), agent, runtime=LocalRuntime(env))

    (request,) = models.requests()
    assert {tool["name"] for tool in request.body["tools"] if tool.get("defer_loading")} == deferred


@pytest.mark.parametrize(
    ("model", "beta", "kind", "sized"),
    [
        ("claude-sonnet-4-6", "computer-use-2025-11-24", "computer_20251124", True),
        ("claude-sonnet-4-5", "computer-use-2025-01-24", "computer_20250124", True),
        ("qwen/qwen3.8-max", None, None, False),
    ],
    ids=["4-6", "4-5", "a-model-without-native-computer-use"],
)
async def test_each_model_gets_the_computer_tool_it_supports_for_the_screen(
    model: str, beta: str | None, kind: str | None, sized: bool, models: Models, tmp_path: Path
) -> None:
    models.script([Turn(text="done")])

    async with fake_screen(width=200, height=100) as screen:
        env = workspace_env(tmp_path / "ws", capabilities=(Capability.rfb(url=screen.url),))
        await rollout(
            Task(env=env.name, id="task"),
            ClaudeAgent(ClaudeConfig(model=model)),
            runtime=LocalRuntime(env),
        )

    (request,) = models.requests()
    (computer,) = [tool for tool in request.body["tools"] if tool["name"] == "computer"]
    assert (request.headers.get("anthropic-beta"), computer.get("type")) == (beta, kind)
    assert (computer.get("display_width_px"), computer.get("display_height_px")) == (
        (200, 100) if sized else (None, None)
    )
    assert sized or "display is 200x100 pixels" in computer["description"]


async def test_an_unknown_editor_command_fails_its_tool_step(
    models: Models, tmp_path: Path
) -> None:
    models.script(
        [call("str_replace_based_edit_tool", command="bogus", path="/x"), Turn(text="done")]
    )

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    (step,) = [step for step in run.trace.steps if isinstance(step, ToolStep)]
    assert step.result is not None
    (text,) = [
        block.text for block in step.result.content if isinstance(block, mcp_types.TextContent)
    ]
    assert (run.trace.status, step.result.isError) == ("completed", True)
    assert "unknown editor command: 'bogus'" in text


INVALID_JSON = Turn(
    native=({"type": "tool_use", "id": "toolu_bad", "name": "bash", "input": '{"command": ls}'},)
)


@pytest.mark.parametrize(
    ("invalid", "status", "requests"),
    [(1, "completed", 2), (2, "completed", 3), (3, "error", 3)],
    ids=["once-is-asked-again", "twice-is-asked-again", "three-times-fails"],
)
async def test_a_call_with_invalid_json_is_sampled_again_until_the_third_fails_the_run(
    invalid: int, status: str, requests: int, models: Models, tmp_path: Path
) -> None:
    models.queue([INVALID_JSON] * invalid + [Turn(text="done")])

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    assert (run.trace.status, len(models.requests())) == (status, requests)


@pytest.mark.parametrize(
    ("errors", "kind", "status", "requests"),
    [
        (1, "overloaded_error", "completed", 2),
        (3, "gateway_timeout", "error", 3),
        (1, "invalid_request_error", "error", 1),
    ],
    ids=["a-transient-error-is-retried", "three-transient-errors-fail", "a-request-error-fails"],
)
async def test_an_error_inside_the_stream_is_retried_only_when_transient(
    errors: int, kind: str, status: str, requests: int, models: Models, tmp_path: Path
) -> None:
    models.queue(
        [stream_error(kind, headers={"retry-after-ms": "0"})] * errors + [Turn(text="done")]
    )

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    assert (run.trace.status, len(models.requests())) == (status, requests)


@pytest.mark.parametrize(
    ("config", "sent"),
    [
        ({}, [["text", "image"], ["text", "image"]]),
        (
            {"max_tool_result_images": 1, "thinking": {"type": "adaptive"}},
            [["text", "text"], ["text", "image"]],
        ),
    ],
    ids=["every-image-by-default", "only-the-newest-image"],
)
async def test_tool_result_images_are_sent_back_up_to_the_configured_count(
    config: dict[str, Any], sent: list[list[str]], models: Models, tmp_path: Path
) -> None:
    screenshots = FastMCP("screenshots")

    @screenshots.tool
    def screenshot() -> ToolResult:
        """Capture the current screen."""
        return ToolResult(
            content=[
                mcp_types.TextContent(type="text", text="screen"),
                mcp_types.ImageContent(
                    type="image", mimeType="image/png", data=base64.b64encode(png(32, 24)).decode()
                ),
            ]
        )

    models.script(
        [
            replace(call("screenshot"), reasoning="first look"),
            replace(call("screenshot"), reasoning="second look"),
            Turn(text="done"),
        ]
    )
    agent = ClaudeAgent(ClaudeConfig.model_validate({"model": MODEL, **config}))

    async with mcp_server(screenshots, name="screenshots") as server:
        env = workspace_env(tmp_path / "ws", capabilities=(server,))
        await rollout(Task(env=env.name, id="task"), agent, runtime=LocalRuntime(env))

    results = models.requests()[-1].body["messages"][2::2]
    assert [[block["type"] for block in m["content"][0]["content"]] for m in results] == sent


@pytest.mark.parametrize(
    ("data", "shape"),
    [
        (png(40, 30), (40, 30)),
        (png(3000, 2000), (3000, 2000)),
        (upright_after_rotation(), (20, 40)),
    ],
    ids=["a-small-png", "a-wide-png", "a-jpeg-turned-upright-by-its-exif"],
)
async def test_an_image_the_editor_views_is_sent_within_claudes_limits_in_its_shape(
    data: bytes, shape: tuple[int, int], models: Models, tmp_path: Path
) -> None:
    root = tmp_path / "ws"
    models.script(
        [
            call("str_replace_based_edit_tool", command="view", path=f"{root}/file"),
            Turn(text="done"),
        ]
    )

    env = workspace_env(root, files={"file": data})
    await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    ((block,),) = [r["content"] for r in models.requests()[-1].body["messages"][-1]["content"]]
    with Image.open(io.BytesIO(base64.b64decode(block["source"]["data"]))) as sent:
        width, height = sent.size
    assert max(width, height) <= 1568 and width * height <= 1_150_000
    assert abs(width / height - shape[0] / shape[1]) < 0.01


async def test_an_image_that_cannot_be_read_is_told_to_the_model_as_an_error(
    models: Models, tmp_path: Path
) -> None:
    """The view itself succeeds; only the image Claude is sent back fails to decode, so
    the error exists only in the tool result the model receives."""
    root = tmp_path / "ws"
    models.script(
        [
            call("str_replace_based_edit_tool", command="view", path=f"{root}/file"),
            Turn(text="done"),
        ]
    )
    corrupt = b"\x89PNG\r\n\x1a\ngarbage"

    env = workspace_env(root, files={"file": corrupt})
    await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    (result,) = models.requests()[-1].body["messages"][-1]["content"]
    assert result["is_error"] is True
    assert "Cannot view image" in result["content"][0]["text"]


async def test_the_newest_user_turn_is_marked_for_prompt_caching(
    models: Models, tmp_path: Path
) -> None:
    models.script([call("bash", command="true"), Turn(text="done")])

    env = workspace_env(tmp_path / "ws")
    await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    first, second = (request.body["messages"] for request in models.requests())
    assert [block.get("cache_control") for block in first[-1]["content"]] == [{"type": "ephemeral"}]
    assert [block.get("cache_control") for block in second[-1]["content"]] == [
        {"type": "ephemeral"}
    ]
    assert "cache_control" not in json.dumps(second[0])


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

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=profile)),
        runtime=LocalRuntime(env),
    )

    (request,) = services.requests("gateway", "POST")
    assert run.trace.content == "from bedrock"
    assert request.path == f"/model/{profile}/invoke"
    assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256")
    assert ("model" in request.json, "stream" in request.json) == (False, False)


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

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

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
    ids=["with-an-explanation", "with-only-a-category"],
)
async def test_a_refusal_lands_on_the_agent_step(
    explanation: str | None, refusal: str, models: Models, tmp_path: Path
) -> None:
    details = {"type": "refusal", "category": "cyber", "explanation": explanation}
    models.script([Turn(finish="refusal", extra={"stop_details": details})])

    env = workspace_env(tmp_path / "ws")
    run = await rollout(
        Task(env=env.name, id="task"),
        ClaudeAgent(ClaudeConfig(model=MODEL)),
        runtime=LocalRuntime(env),
    )

    (step,) = [step for step in steps(run.trace_id) if step["source"] == "agent"]
    assert (step["finish_reason"], step["refusal"]) == ("refusal", refusal)
