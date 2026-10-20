"""``ClaudeAgent`` — ``get_response`` parsing over a fake streaming Messages client,
plus the pure ``_citation`` / ``_cache_last_user_block`` helpers.
"""

from __future__ import annotations

import base64
import json
import random
import struct
import zlib
from contextlib import AsyncExitStack
from io import BytesIO
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, Mock, call

import fastmcp
import httpx
import httpx2
import mcp.types as mcp_types
import pytest
from anthropic import APIStatusError, AsyncAnthropic, Omit
from fastmcp.tools import ToolResult
from mcp.types import ImageContent
from PIL import Image
from pydantic import ValidationError

from hud.agents.claude.agent import ClaudeAgent
from hud.agents.claude.tools.computer import ClaudeComputerTool
from hud.agents.tool_agent import RunState
from hud.agents.types import ClaudeConfig, ToolStep
from hud.capabilities import Capability, MCPClient, RFBClient
from hud.settings import settings
from hud.types import MCPToolCall, MCPToolResult, Trace

if TYPE_CHECKING:
    from hud.eval.run import Run


class FakeStream:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *_a: Any) -> bool:
        return False

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        return self._outcome


class FakeMessages:
    def __init__(self, *outcomes: Any) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> FakeStream:
        self.requests.append(kwargs)
        outcome = self._outcomes[self.calls]
        self.calls += 1
        return FakeStream(outcome)


class FakeAnthropic:
    def __init__(self, *outcomes: Any) -> None:
        self.beta = SimpleNamespace(messages=FakeMessages(*outcomes))


def _final(*content: Any, stop_reason: str, stop_details: Any = None) -> Any:
    """A fake ``BetaMessage``: content blocks plus the always-present envelope."""
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        stop_details=stop_details,
        model="claude-test-v9",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, cache_read_input_tokens=3),
    )


def _agent(*outcomes: Any) -> ClaudeAgent:
    from hud.agents.types import ClaudeConfig

    return ClaudeAgent(
        ClaudeConfig(model="claude-test", max_tokens=1024, model_client=FakeAnthropic(*outcomes))
    )


def _state(agent: ClaudeAgent) -> Any:
    from hud.agents.tool_agent import RunState

    return RunState(messages=[agent._format_message("user", "go")])


def _inline_error(type_: str, *, headers: dict[str, str] | None = None) -> APIStatusError:
    body = {"type": "error", "error": {"type": type_, "message": "stream failed"}}
    response = httpx.Response(
        200,
        headers=headers,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return APIStatusError(str(body), response=response, body=body)


def test_format_message_shape() -> None:
    agent = _agent(SimpleNamespace(content=[], stop_reason="end_turn"))
    msg = agent._format_message("assistant", "hi")
    assert msg["role"] == "assistant"


async def test_get_response_text_and_tool_use() -> None:
    final = _final(
        SimpleNamespace(type="text", text="hello", citations=None),
        SimpleNamespace(type="tool_use", id="t1", name="bash", input={"command": "ls"}),
        stop_reason="tool_use",
    )
    agent = _agent(final)
    state = _state(agent)

    result = await agent.get_response(state)

    assert result.content == "hello"
    assert [tc.name for tc in result.tool_calls] == ["bash"]
    assert result.tool_calls[0].arguments == {"command": "ls"}
    assert result.done is False
    assert result.finish_reason == "tool_use"
    # Model and usage are normalized off the provider response.
    assert result.model == "claude-test-v9"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert result.usage.cached_tokens == 3


async def test_get_response_done_on_text_only() -> None:
    final = _final(
        SimpleNamespace(type="text", text="done", citations=None),
        stop_reason="end_turn",
    )
    agent = _agent(final)
    result = await agent.get_response(_state(agent))
    assert result.done is True
    assert result.content == "done"
    assert result.tool_calls == []
    assert result.refusal is None


async def test_get_response_surfaces_refusal_explanation() -> None:
    explanation = (
        "This request triggered restrictions on violative cyber content and was "
        "blocked under Anthropic's Usage Policy."
    )
    final = _final(
        stop_reason="refusal",
        stop_details=SimpleNamespace(
            type="refusal",
            category="cyber",
            explanation=explanation,
        ),
    )
    agent = _agent(final)
    result = await agent.get_response(_state(agent))

    assert result.finish_reason == "refusal"
    assert result.refusal == explanation
    assert result.done is True
    assert result.tool_calls == []


async def test_get_response_surfaces_refusal_category_when_explanation_missing() -> None:
    final = _final(
        stop_reason="refusal",
        stop_details=SimpleNamespace(
            type="refusal",
            category="cyber",
            explanation=None,
        ),
    )
    agent = _agent(final)
    result = await agent.get_response(_state(agent))

    assert result.finish_reason == "refusal"
    assert result.refusal is not None
    assert "cyber" in result.refusal
    assert result.done is True
    assert result.tool_calls == []


async def test_get_response_collects_thinking() -> None:
    final = _final(
        SimpleNamespace(type="thinking", thinking="pondering"),
        SimpleNamespace(type="text", text="answer", citations=None),
        stop_reason="end_turn",
    )
    agent = _agent(final)
    result = await agent.get_response(_state(agent))
    assert result.reasoning == "pondering"


@pytest.mark.parametrize("error_type", ["upstream_error", "timeout_error"])
async def test_get_response_retries_transient_inline_stream_error(
    monkeypatch: pytest.MonkeyPatch,
    error_type: str,
) -> None:
    final = _final(
        SimpleNamespace(type="text", text="recovered", citations=None),
        stop_reason="end_turn",
    )
    agent = _agent(_inline_error(error_type), _inline_error(error_type), final)
    state = _state(agent)
    sleep = AsyncMock()
    jitter = Mock(side_effect=[0.25, 1.5])
    monkeypatch.setattr("hud.agents.claude.agent.asyncio.sleep", sleep)
    monkeypatch.setattr("hud.agents.claude.agent._STREAM_JITTER.uniform", jitter)

    result = await agent.get_response(state)

    messages = cast("FakeMessages", cast("Any", agent.anthropic_client).beta.messages)
    assert messages.calls == 3
    assert result.content == "recovered"
    assert len(state.messages) == 2
    assert sleep.await_args_list == [call(0.25), call(1.5)]
    assert jitter.call_args_list == [call(0.0, 1.0), call(0.0, 2.0)]


async def test_get_response_raises_after_transient_stream_retry_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(
        _inline_error("gateway_timeout"),
        _inline_error("gateway_timeout"),
        _inline_error("gateway_timeout"),
    )
    state = _state(agent)
    sleep = AsyncMock()
    monkeypatch.setattr("hud.agents.claude.agent.asyncio.sleep", sleep)
    monkeypatch.setattr("hud.agents.claude.agent._STREAM_JITTER.uniform", Mock(return_value=0.0))

    with pytest.raises(APIStatusError, match="gateway_timeout"):
        await agent.get_response(state)

    messages = cast("FakeMessages", cast("Any", agent.anthropic_client).beta.messages)
    assert messages.calls == 3
    assert len(state.messages) == 1
    assert sleep.await_count == 2


@pytest.mark.parametrize(
    ("headers", "expected_delay"),
    [
        ({"retry-after": "1.75"}, 1.75),
        ({"retry-after-ms": "1750"}, 1.75),
        ({"retry-after": "60"}, 5.0),
    ],
    ids=["seconds", "milliseconds", "capped"],
)
async def test_get_response_honors_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    expected_delay: float,
) -> None:
    final = _final(
        SimpleNamespace(type="text", text="recovered", citations=None),
        stop_reason="end_turn",
    )
    agent = _agent(_inline_error("overloaded_error", headers=headers), final)
    sleep = AsyncMock()
    jitter = Mock()
    monkeypatch.setattr("hud.agents.claude.agent.asyncio.sleep", sleep)
    monkeypatch.setattr("hud.agents.claude.agent._STREAM_JITTER.uniform", jitter)

    result = await agent.get_response(_state(agent))

    assert result.content == "recovered"
    sleep.assert_awaited_once_with(expected_delay)
    jitter.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadError(
            "stream interrupted",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        httpx.ReadTimeout(
            "stream interrupted",
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        httpx2.ReadError(
            "stream interrupted",
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
        httpx2.ReadTimeout(
            "stream interrupted",
            request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
        ),
    ],
    ids=["httpx-read-error", "httpx-read-timeout", "httpx2-read-error", "httpx2-read-timeout"],
)
async def test_get_response_retries_interrupted_stream(
    monkeypatch: pytest.MonkeyPatch,
    error: httpx.TransportError | httpx2.TransportError,
) -> None:
    final = _final(
        SimpleNamespace(type="text", text="recovered", citations=None),
        stop_reason="end_turn",
    )
    agent = _agent(error, final)
    state = _state(agent)
    monkeypatch.setattr("hud.agents.claude.agent.asyncio.sleep", AsyncMock())

    result = await agent.get_response(state)

    messages = cast("FakeMessages", cast("Any", agent.anthropic_client).beta.messages)
    assert messages.calls == 2
    assert result.content == "recovered"
    assert len(state.messages) == 2


async def test_get_response_does_not_retry_non_transient_inline_stream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(_inline_error("invalid_request_error"))
    state = _state(agent)
    sleep = AsyncMock()
    monkeypatch.setattr("hud.agents.claude.agent.asyncio.sleep", sleep)

    with pytest.raises(APIStatusError, match="invalid_request_error"):
        await agent.get_response(state)

    messages = cast("FakeMessages", cast("Any", agent.anthropic_client).beta.messages)
    assert messages.calls == 1
    assert len(state.messages) == 1
    sleep.assert_not_awaited()


def test_citation_char_location() -> None:
    raw = SimpleNamespace(
        type="char_location",
        cited_text="quote",
        document_index=2,
        document_title="doc",
        start_char_index=0,
        end_char_index=5,
    )
    cit = ClaudeAgent._citation(cast("Any", raw))
    assert cit.type == "document_citation"
    assert cit.source == "2"
    assert cit.start_index == 0


def test_cache_last_user_block_marks_content() -> None:
    agent = _agent(SimpleNamespace(content=[], stop_reason="end_turn"))
    messages = [agent._format_message("user", "hi")]
    out = ClaudeAgent._cache_last_user_block(messages)
    content = cast("list[Any]", out[-1]["content"])
    block = cast("dict[str, Any]", content[0])
    assert block.get("cache_control") == {"type": "ephemeral"}


def _image_result(data: bytes, *, computer: bool = False) -> Any:
    agent = _agent()
    state = RunState()
    if computer:
        spec = ClaudeComputerTool.default_spec("claude")
        assert spec is not None
        tool = ClaudeComputerTool(spec=spec, client=Mock(spec=RFBClient))
        state.tools[tool.provider_name] = tool
        name = tool.provider_name
    else:
        name = "read_file"
    content = ImageContent(type="image", mimeType="image/png", data=base64.b64encode(data).decode())
    result = MCPToolResult(content=[content])
    original = result.model_dump()
    message = agent._format_result(
        MCPToolCall(id="image-call", name=name, arguments={}), result, state
    )
    assert result.model_dump() == original
    assert isinstance(message, dict)
    return next(iter(message["content"]))


@pytest.mark.parametrize("size", [(4000, 2000), (2000, 4000), (2000, 2000), (32, 24)])
def test_tool_images_fit_claude_limits_without_upscaling(size: tuple[int, int]) -> None:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    result = _image_result(buffer.getvalue())
    assert not result["is_error"]
    source = result["content"][0]["source"]
    assert len(source["data"]) <= 5_000_000
    with Image.open(BytesIO(base64.b64decode(source["data"]))) as image:
        assert max(image.size) <= 1568
        assert image.width * image.height <= 1_150_000
        assert image.width <= size[0] and image.height <= size[1]
        assert image.width / image.height == pytest.approx(size[0] / size[1], rel=0.005)


def test_claude_compresses_noisy_transparent_tool_images() -> None:
    size = (1072, 1072)
    image = Image.frombytes("RGBA", size, random.Random(0).randbytes(size[0] * size[1] * 4))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    assert len(base64.b64encode(buffer.getvalue())) > 5_000_000
    result = _image_result(buffer.getvalue())
    assert not result["is_error"]
    source = result["content"][0]["source"]
    assert source["media_type"] == "image/jpeg"
    assert len(source["data"]) <= 5_000_000
    with Image.open(BytesIO(base64.b64decode(source["data"]))) as viewed:
        assert viewed.size == size
        assert viewed.mode == "RGB"


def test_claude_applies_image_orientation() -> None:
    image = Image.new("RGB", (40, 20), "red")
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    source = _image_result(buffer.getvalue())["content"][0]["source"]
    with Image.open(BytesIO(base64.b64decode(source["data"]))) as viewed:
        assert viewed.size == (20, 40)
        assert viewed.getexif().get(274, 1) == 1


def test_claude_preserves_small_png_transparency() -> None:
    buffer = BytesIO()
    Image.new("RGBA", (32, 24), (255, 0, 0, 128)).save(buffer, format="PNG")
    source = _image_result(buffer.getvalue())["content"][0]["source"]
    assert source["media_type"] == "image/png"
    with Image.open(BytesIO(base64.b64decode(source["data"]))) as viewed:
        assert viewed.getpixel((0, 0)) == (255, 0, 0, 128)


@pytest.mark.filterwarnings("ignore:Image size.*:PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize("size", [(4001, 4000), (10000, 10000)])
def test_claude_rejects_oversized_source_before_decoding(size: tuple[int, int]) -> None:
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    data = bytearray(buffer.getvalue())
    data[16:24] = struct.pack(">II", *size)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    result = _image_result(bytes(data))
    assert result["is_error"]
    assert result["content"][0]["type"] == "text"
    assert "16,000,000-pixel source limit" in result["content"][0]["text"]


def test_claude_returns_invalid_image_as_tool_error() -> None:
    result = _image_result(b"invalid image")
    assert result["is_error"]
    assert result["content"][0]["type"] == "text"
    assert "Cannot view image" in result["content"][0]["text"]


def test_claude_preserves_computer_screenshot_coordinate_space() -> None:
    buffer = BytesIO()
    Image.new("RGB", (2000, 1000), "white").save(buffer, format="PNG")
    data = buffer.getvalue()
    result = _image_result(data, computer=True)
    assert not result["is_error"]
    source = result["content"][0]["source"]
    assert source["media_type"] == "image/png"
    assert base64.b64decode(source["data"]) == data


@pytest.mark.parametrize("limit", [None, 1])
async def test_image_history_preserves_tool_results_and_thinking(
    limit: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "telemetry_enabled", False)
    buffer = BytesIO()
    Image.new("RGB", (32, 24), "white").save(buffer, format="PNG")
    image = ImageContent(
        type="image", mimeType="image/png", data=base64.b64encode(buffer.getvalue()).decode()
    )
    server = fastmcp.FastMCP("screenshots")

    @server.tool
    def screenshot() -> ToolResult:
        """Capture the current screen."""
        return ToolResult(content=[mcp_types.TextContent(type="text", text="screen"), image])

    client = FakeAnthropic(
        *(
            _final(
                SimpleNamespace(
                    type="thinking", thinking=f"reason-{turn}", signature=f"sig-{turn}"
                ),
                SimpleNamespace(type="tool_use", id=f"shot-{turn}", name="screenshot", input={}),
                stop_reason="tool_use",
            )
            for turn in (1, 2)
        ),
        _final(SimpleNamespace(type="text", text="done", citations=None), stop_reason="end_turn"),
    )
    config = ClaudeConfig(
        model="claude-test",
        model_client=client,
        max_steps=3,
        max_tool_result_images=limit,
        thinking={"type": "adaptive"} if limit is not None else None,
    )
    restored = ClaudeConfig.model_validate(config.model_dump(mode="json"))
    assert restored.max_tool_result_images == limit
    assert restored.thinking == config.thinking
    async with fastmcp.Client(server) as mcp_client:
        cap = Capability.mcp(name="screenshots", url="https://tools.invalid/mcp")
        connection = MCPClient(cap, mcp_client, AsyncExitStack())
        trace = Trace()
        run = cast(
            "Run",
            SimpleNamespace(
                trace=trace,
                record=trace.record,
                client=SimpleNamespace(
                    manifest=SimpleNamespace(bindings=[cap]),
                    open=AsyncMock(return_value=connection),
                ),
                prompt_messages=[
                    mcp_types.PromptMessage(
                        role="user", content=mcp_types.TextContent(type="text", text="go")
                    )
                ],
            ),
        )
        await ClaudeAgent(config)(run)

    assert trace.status == "completed"
    requests = client.beta.messages.requests
    assert len(requests) == 3
    for request in requests:
        if limit is None:
            assert isinstance(request.get("thinking", Omit()), Omit)
            assert isinstance(request.get("betas", Omit()), Omit)
        else:
            assert "thinking-binding-controls-2026-08-01" in request["betas"]
            assert request["thinking"] == {
                "type": "adaptive",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
    messages = requests[-1]["messages"]
    results = [message["content"][0] for message in messages[2::2]]
    assert [result["tool_use_id"] for result in results] == ["shot-1", "shot-2"]
    assert [result["content"][0]["text"] for result in results] == ["screen", "screen"]
    assert [result["content"][1]["type"] for result in results] == (
        ["image", "image"] if limit is None else ["text", "image"]
    )
    assert [
        (message["content"][0].thinking, message["content"][0].signature)
        for message in messages[1::2]
    ] == [("reason-1", "sig-1"), ("reason-2", "sig-2")]
    for step in trace.steps:
        if isinstance(step, ToolStep):
            assert step.result is not None
            recorded_image = step.result.content[1]
            assert isinstance(recorded_image, ImageContent)
            assert recorded_image.data == image.data


@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}])
def test_image_history_requires_enabled_thinking(thinking: dict[str, str] | None) -> None:
    with pytest.raises(ValidationError, match="requires explicit adaptive or enabled thinking"):
        ClaudeConfig.model_validate({"max_tool_result_images": 1, "thinking": thinking})


@pytest.mark.parametrize(
    ("max_tokens", "budget_tokens"),
    [(16384, 16384), (16384, 16385), (2048, 2048), (2048, 2049)],
)
def test_enabled_thinking_budget_must_be_below_max_tokens(
    max_tokens: int, budget_tokens: int
) -> None:
    with pytest.raises(ValidationError, match="budget_tokens must be less than max_tokens"):
        ClaudeConfig(
            max_tokens=max_tokens, thinking={"type": "enabled", "budget_tokens": budget_tokens}
        )


@pytest.mark.parametrize("max_tokens", [1025, 2048])
def test_enabled_thinking_budget_below_max_tokens_is_allowed(max_tokens: int) -> None:
    config = ClaudeConfig(
        max_tokens=max_tokens, thinking={"type": "enabled", "budget_tokens": max_tokens - 1}
    )
    assert config.thinking == {"type": "enabled", "budget_tokens": max_tokens - 1}


def _computer_member(id: str, name: str, **arguments: Any) -> Any:
    return SimpleNamespace(
        type="tool_use", id=id, name=name, toolset_name="computer", input=arguments
    )


async def test_sonnet55_computer_batch_round_trips_namespaces_and_stops_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "telemetry_enabled", False)
    server = fastmcp.FastMCP("custom")
    custom_calls: list[str] = []

    @server.tool
    def key() -> str:
        """A custom operation that shares a computer member's name."""
        custom_calls.append("key")
        return "custom key"

    client = FakeAnthropic(
        _final(
            _computer_member("keys", "key", text="ctrl+a", repeat=2),
            SimpleNamespace(type="tool_use", id="custom", name="key", input={}),
            _computer_member("failed", "key"),
            _computer_member("skipped", "left_click", coordinate=[10, 20]),
            SimpleNamespace(type="tool_use", id="skipped-custom", name="key", input={}),
            stop_reason="tool_use",
        ),
        _final(_computer_member("retry", "key", text="Tab"), stop_reason="tool_use"),
        _final(SimpleNamespace(type="text", text="done", citations=None), stop_reason="end_turn"),
    )
    keyboard = Mock()
    connection = SimpleNamespace(
        keyboard=keyboard,
        mouse=Mock(),
        video=SimpleNamespace(width=1280, height=720),
        drain=AsyncMock(),
    )
    screen = Capability.rfb(name="screen", url="rfb://screen.invalid:5900")
    rfb = RFBClient(screen, cast("Any", connection), AsyncExitStack())
    custom = Capability.mcp(name="custom", url="https://tools.invalid/mcp")
    async with fastmcp.Client(server) as mcp_client:
        mcp = MCPClient(custom, mcp_client, AsyncExitStack())
        trace = Trace()
        run = cast(
            "Run",
            SimpleNamespace(
                trace=trace,
                record=trace.record,
                client=SimpleNamespace(
                    manifest=SimpleNamespace(bindings=[screen, custom]),
                    open=AsyncMock(side_effect=lambda name: rfb if name == "screen" else mcp),
                ),
                prompt_messages=[
                    mcp_types.PromptMessage(
                        role="user", content=mcp_types.TextContent(type="text", text="go")
                    )
                ],
            ),
        )
        await ClaudeAgent(
            ClaudeConfig(model="claude-sonnet-5-5", model_client=client, max_steps=3)
        )(run)

    assert trace.status == "completed"
    assert trace.content == "done"
    assert custom_calls == ["key"]
    assert keyboard.press.call_args_list == [
        call("Control_L", "a"),
        call("Control_L", "a"),
        call("Tab"),
    ]
    connection.mouse.click.assert_not_called()
    requests = client.beta.messages.requests
    assert requests[0]["tools"][0] == {"type": "computer_toolset_20260801"}
    assert isinstance(requests[0]["betas"], Omit)
    messages = requests[1]["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    results = messages[-1]["content"]
    assert sum("cache_control" in block for block in results) == 1
    assert "cache_control" in results[-1]
    assert [block["tool_use_id"] for block in results] == [
        "keys",
        "custom",
        "failed",
        "skipped",
        "skipped-custom",
    ]
    assert [block.get("toolset_name") for block in results] == [
        "computer",
        None,
        "computer",
        "computer",
        None,
    ]
    assert [block["is_error"] for block in results] == [False, False, True, True, True]
    assert results[-2]["content"][0]["text"] == (
        "Not executed: an earlier computer action in this turn failed."
    )
    recorded_calls = [step.call for step in trace.steps if isinstance(step, ToolStep)]
    assert recorded_calls[0] is not None
    assert recorded_calls[0].name == "computer"
    assert recorded_calls[0].provider_name == "key"
    assert recorded_calls[0].toolset_name == "computer"
    assert recorded_calls[1] is not None
    assert recorded_calls[1].name == "key"
    assert recorded_calls[1].toolset_name is None
    assert requests[2]["messages"][-1]["content"][0]["is_error"] is False


@pytest.mark.parametrize(
    ("model", "expected_type", "expected_beta"),
    [
        ("claude-sonnet-5-5", "computer_toolset_20260801", None),
        ("claude-sonnet-4-6", "computer_20251124", "computer-use-2025-11-24"),
        ("claude-sonnet-4-5", "computer_20250124", "computer-use-2025-01-24"),
        ("qwen/qwen3.8-max", None, None),
    ],
)
async def test_computer_schema_selection_preserves_legacy_and_generic_models(
    model: str, expected_type: str | None, expected_beta: str | None
) -> None:
    agent = ClaudeAgent(ClaudeConfig(model=model, model_client=FakeAnthropic()))
    screen = Mock(spec=RFBClient)
    screen.width, screen.height = 1280, 720
    tools, params = await agent._build_tools({"screen": screen})
    assert params[0].get("type") == expected_type
    assert getattr(tools["computer"].spec, "beta", None) == expected_beta
    if expected_type != "computer_toolset_20260801":
        assert params[0]["name"] == "computer"


async def test_sonnet55_bedrock_uses_its_supported_legacy_computer_protocol() -> None:
    from anthropic import AsyncAnthropicBedrock

    client = AsyncAnthropicBedrock(
        aws_access_key="test", aws_secret_key="test", aws_region="us-east-1"
    )
    try:
        agent = ClaudeAgent(
            ClaudeConfig(model="anthropic.claude-sonnet-5-5-v1:0", model_client=client)
        )
        screen = Mock(spec=RFBClient)
        screen.width, screen.height = 1280, 720
        tools, params = await agent._build_tools({"screen": screen})
        assert params[0]["type"] == "computer_20251124"
        assert getattr(tools["computer"].spec, "beta", None) == "computer-use-2025-11-24"
    finally:
        await client.close()


async def test_sonnet55_toolset_survives_the_real_anthropic_wire_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "telemetry_enabled", False)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        tool_turn = len(requests) == 1
        block = (
            {
                "type": "tool_use",
                "id": "native-shot",
                "name": "screenshot",
                "toolset_name": "computer",
                "input": {},
            }
            if tool_turn
            else {"type": "text", "text": "done"}
        )
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": block},
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use" if tool_turn else "end_turn"},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    source = BytesIO()
    Image.new("RGB", (128, 72), "white").save(source, format="PNG")
    screen = Capability.rfb(name="screen", url="rfb://screen.invalid:5900")
    rfb = Mock(spec=RFBClient)
    rfb.width, rfb.height = 128, 72
    rfb.screenshot_png = AsyncMock(return_value=(source.getvalue(), "image/png"))
    trace = Trace()
    run = cast(
        "Run",
        SimpleNamespace(
            trace=trace,
            record=trace.record,
            client=SimpleNamespace(
                manifest=SimpleNamespace(bindings=[screen]),
                open=AsyncMock(return_value=rfb),
            ),
            prompt_messages=[
                mcp_types.PromptMessage(
                    role="user", content=mcp_types.TextContent(type="text", text="go")
                )
            ],
        ),
    )
    async with AsyncAnthropic(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
    ) as client:
        await ClaudeAgent(
            ClaudeConfig(model="claude-sonnet-5-5", model_client=client, max_steps=2)
        )(run)
    assert trace.status == "completed"
    assert len(requests) == 2
    assert "anthropic-beta" not in requests[0].headers
    assert json.loads(requests[0].content)["tools"] == [{"type": "computer_toolset_20260801"}]
    result = json.loads(requests[1].content)["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "native-shot"
    assert result["toolset_name"] == "computer"
    assert result["content"][0]["type"] == "image"
    assert not result["is_error"]


async def test_computer_toolset_rejects_reserved_custom_computer_name() -> None:
    agent = ClaudeAgent(ClaudeConfig(model="claude-sonnet-5-5", model_client=FakeAnthropic()))
    screen = Mock(spec=RFBClient)
    screen.width, screen.height = 1280, 720
    custom = Mock(spec=MCPClient)
    custom.list_tools = AsyncMock(
        return_value=[mcp_types.Tool(name="computer", inputSchema={"type": "object"})]
    )
    with pytest.raises(ValueError, match=r"MCP tool name collision.*computer"):
        await agent._build_tools({"screen": screen, "custom": custom})
