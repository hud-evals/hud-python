"""``OpenAIChatAgent`` — chat.completions ``get_response`` parsing + error path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletionChunk
from pydantic import ValidationError

from hud.agents.openai_compatible.agent import OpenAIChatAgent, OpenAIChatRunState
from hud.agents.types import OpenAIChatConfig


class FakeStream:
    def __init__(self, chunks: list[ChatCompletionChunk]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def __aiter__(self) -> Any:
        for chunk in self._chunks:
            yield chunk


class FakeCompletions:
    def __init__(self, response: Any, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self._error is not None:
            raise self._error
        if kwargs["stream"]:
            return FakeStream(self._response)
        return self._response


class FakeOpenAI:
    def __init__(self, response: Any, error: Exception | None = None) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(response, error))


def _agent(response: Any, error: Exception | None = None, **config: Any) -> OpenAIChatAgent:
    client = cast("Any", FakeOpenAI(response, error))
    return OpenAIChatAgent(OpenAIChatConfig(model="m", model_client=client, **config))


def _requests(agent: OpenAIChatAgent) -> list[dict[str, Any]]:
    return cast("Any", agent.oai).chat.completions.requests


def _chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish_reason: str | None = None,
    usage: tuple[int, int] | None = None,
) -> ChatCompletionChunk:
    data: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m-v1",
        "choices": []
        if delta is None
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        prompt_tokens, completion_tokens = usage
        data["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return ChatCompletionChunk.model_validate(data)


def _response(content: str, tool_calls: list[Any]) -> Any:
    dump: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        dump["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        refusal=None,
        model_dump=lambda exclude_none=True: dump,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    return SimpleNamespace(
        choices=[choice],
        model="m-v1",
        usage=SimpleNamespace(prompt_tokens=6, completion_tokens=2, prompt_tokens_details=None),
    )


def _state(agent: OpenAIChatAgent) -> Any:
    return OpenAIChatRunState(messages=[agent._format_message("user", "go")])


async def test_get_response_text_only() -> None:
    agent = _agent(_response("hi", []))
    result = await agent.get_response(_state(agent))
    assert result.content == "hi"
    assert result.done is True
    assert result.tool_calls == []
    # Model and usage are normalized off the provider response.
    assert result.model == "m-v1"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 6
    assert result.usage.completion_tokens == 2


async def test_get_response_with_tool_call() -> None:
    tc = SimpleNamespace(
        type="function",
        id="c1",
        function=SimpleNamespace(name="read", arguments='{"path": "x"}'),
    )
    agent = _agent(_response("", [tc]))
    result = await agent.get_response(_state(agent))
    assert [c.name for c in result.tool_calls] == ["read"]
    assert result.tool_calls[0].arguments == {"path": "x"}
    assert result.done is False


async def test_get_response_error_path() -> None:
    agent = _agent(None, error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await agent.get_response(_state(agent))


async def test_get_response_malformed_tool_args() -> None:
    """Unparseable arguments (e.g. truncated at the token limit) yield a call carrying
    the raw string, and the recorded message replays "{}" so history stays parseable."""
    tc = SimpleNamespace(
        type="function",
        id="c1",
        function=SimpleNamespace(name="bash", arguments='{"command": "'),
    )
    agent = _agent(_response("", [tc]))
    state = _state(agent)
    result = await agent.get_response(state)
    assert result.error is None
    assert result.done is False
    (call,) = result.tool_calls
    assert call.id == "c1"
    assert call.arguments == '{"command": "'
    recorded = state.messages[-1]
    assert recorded["tool_calls"][0]["function"]["arguments"] == "{}"


async def test_default_request_is_not_streamed() -> None:
    agent = _agent(_response("hi", []))
    await agent.get_response(_state(agent))
    (request,) = _requests(agent)
    assert request["stream"] is False
    assert "stream_options" not in request


async def test_streamed_response_is_accumulated() -> None:
    chunks = [
        _chunk({"role": "assistant", "content": "Let me "}),
        _chunk({"content": "look."}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"pa'},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'th": "x"}'}}]}),
        _chunk({}, finish_reason="tool_calls"),
        _chunk(usage=(6, 2)),
    ]
    agent = _agent(
        chunks,
        stream=True,
        completion_kwargs={"stream_options": {"continuous_usage_stats": True}},
    )
    state = _state(agent)
    result = await agent.get_response(state)

    (request,) = _requests(agent)
    assert request["stream"] is True
    assert request["stream_options"] == {"continuous_usage_stats": True, "include_usage": True}
    assert result.content == "Let me look."
    assert result.finish_reason == "tool_calls"
    assert result.model == "m-v1"
    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [
        ("c1", "read", {"path": "x"})
    ]
    assert result.done is False
    assert result.usage is not None
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (6, 2)
    recorded = state.messages[-1]
    assert recorded["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "x"}'}}
    ]


async def test_streamed_reasoning_fields_are_accumulated() -> None:
    """OpenRouter repeats a detail's ``type``/``format`` on every chunk and sends
    encrypted entries whole; only the text pieces concatenate."""
    detail = {"type": "reasoning.text", "format": "unknown", "index": 0}
    encrypted = {"type": "reasoning.encrypted", "format": "unknown", "index": 1}
    chunks = [
        _chunk(
            {
                "role": "assistant",
                "reasoning_content": "Think ",
                "reasoning_details": [{**detail, "text": "Think "}],
            }
        ),
        _chunk(
            {
                "reasoning_content": "hard.",
                "reasoning_details": [{**detail, "text": "hard."}],
            }
        ),
        _chunk({"reasoning_details": [{**encrypted, "id": "rs_1", "data": "blob"}]}),
        _chunk({"content": "done"}, finish_reason="stop"),
    ]
    agent = _agent(chunks, stream=True)
    state = _state(agent)
    result = await agent.get_response(state)

    assert result.reasoning == "Think hard."
    assert result.content == "done"
    assert result.usage is None
    recorded = state.messages[-1]
    assert recorded["reasoning_content"] == "Think hard."
    assert recorded["reasoning_details"] == [
        {**detail, "text": "Think hard."},
        {**encrypted, "id": "rs_1", "data": "blob"},
    ]


async def test_empty_stream_raises() -> None:
    agent = _agent([], stream=True)
    with pytest.raises(RuntimeError, match="without any chunks"):
        await agent.get_response(_state(agent))


def test_stream_rejects_token_capture() -> None:
    with pytest.raises(ValidationError, match="return_token_ids"):
        OpenAIChatConfig(
            model="m",
            stream=True,
            completion_kwargs={"extra_body": {"return_token_ids": True}},
        )
