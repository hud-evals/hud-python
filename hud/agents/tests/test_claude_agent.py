"""``ClaudeAgent`` — ``get_response`` parsing over a fake streaming Messages client,
plus the pure ``_citation`` / ``_cache_last_user_block`` helpers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import httpx2
import pytest
from anthropic import APIStatusError

from hud.agents.claude.agent import ClaudeAgent


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


def _final(*content: Any, stop_reason: str) -> Any:
    """A fake ``BetaMessage``: content blocks plus the always-present envelope."""
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
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


def _inline_error(type_: str) -> APIStatusError:
    body = {"type": "error", "error": {"type": type_, "message": "stream failed"}}
    response = httpx.Response(
        200,
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


async def test_get_response_collects_thinking() -> None:
    final = _final(
        SimpleNamespace(type="thinking", thinking="pondering"),
        SimpleNamespace(type="text", text="answer", citations=None),
        stop_reason="end_turn",
    )
    agent = _agent(final)
    result = await agent.get_response(_state(agent))
    assert result.reasoning == "pondering"


async def test_get_response_retries_transient_inline_stream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = _final(
        SimpleNamespace(type="text", text="recovered", citations=None),
        stop_reason="end_turn",
    )
    agent = _agent(_inline_error("upstream_error"), final)
    state = _state(agent)
    sleep = AsyncMock()
    monkeypatch.setattr("hud.agents.claude.agent.asyncio.sleep", sleep)

    result = await agent.get_response(state)

    messages = cast("FakeMessages", cast("Any", agent.anthropic_client).beta.messages)
    assert messages.calls == 2
    assert result.content == "recovered"
    assert len(state.messages) == 2
    sleep.assert_awaited_once_with(1.0)


async def test_get_response_raises_after_transient_stream_retry_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _agent(_inline_error("gateway_timeout"), _inline_error("gateway_timeout"))
    state = _state(agent)
    monkeypatch.setattr("hud.agents.claude.agent.asyncio.sleep", AsyncMock())

    with pytest.raises(APIStatusError, match="gateway_timeout"):
        await agent.get_response(state)

    messages = cast("FakeMessages", cast("Any", agent.anthropic_client).beta.messages)
    assert messages.calls == 2
    assert len(state.messages) == 1


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
