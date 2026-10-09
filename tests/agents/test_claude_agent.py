"""``ClaudeAgent`` — ``get_response`` parsing over a fake streaming Messages client,
plus the pure ``_citation`` / ``_cache_last_user_block`` helpers.
"""

from __future__ import annotations

import base64
import random
import struct
import zlib
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, call

import httpx
import httpx2
import pytest
from anthropic import APIStatusError
from mcp.types import ImageContent
from PIL import Image

from hud.agents.claude.agent import ClaudeAgent
from hud.agents.claude.tools.computer import ClaudeComputerTool
from hud.agents.tool_agent import RunState
from hud.capabilities import RFBClient
from hud.types import MCPToolCall, MCPToolResult


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

    def stream(self, **_kwargs: Any) -> FakeStream:
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
