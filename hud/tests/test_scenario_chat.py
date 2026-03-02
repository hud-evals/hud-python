from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from hud.scenario_chat import ChatEvent, ScenarioChatSession, run_scenario_chat_interactive
from hud.tools.types import EvaluationResult


class FakeCtx:
    def __init__(self) -> None:
        self.prompt = "Solve the task"
        self.system_prompt: str | None = None
        self.trace_id = "trace-123"
        self.reward: float | None = None
        self.evaluation_result: EvaluationResult | None = None
        self.submitted: str | None = None
        self.tool_calls_seen: list[Any] = []

    def as_openai_chat_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def as_openai_responses_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    async def call_tool(self, call: Any, /, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        self.tool_calls_seen.append(call)

        if isinstance(call, dict):
            call_id = call.get("id", "unknown")
            return {"role": "tool", "tool_call_id": call_id, "content": "ok"}

        if getattr(call, "type", None) == "function_call":
            return {"type": "function_call_output", "call_id": call.id, "output": "ok"}

        return {"role": "tool", "tool_call_id": call.id, "content": "ok"}

    async def submit(self, answer: str) -> None:
        self.submitted = answer


class _SequentialApi:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self._idx = 0
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self._responses[self._idx]
        self._idx += 1
        return response


class _AsyncChunkIter:
    """Async iterator over a list of chunk objects (simulates OpenAI streaming)."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self._idx = 0

    def __aiter__(self) -> _AsyncChunkIter:
        return self

    async def __anext__(self) -> Any:
        if self._idx >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


def _make_stream_chunk(
    *, content: str | None = None, tool_calls: list[Any] | None = None
) -> Any:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def _build_stream_response(
    *, content: str = "", tool_calls: list[dict[str, Any]] | None = None
) -> _AsyncChunkIter:
    """Build a mock streaming response (async iterable of chunks)."""
    chunks: list[Any] = []
    if content:
        for char in content:
            chunks.append(_make_stream_chunk(content=char))

    if tool_calls:
        for i, tc in enumerate(tool_calls):
            chunks.append(_make_stream_chunk(tool_calls=[
                SimpleNamespace(
                    index=i,
                    id=tc["id"],
                    function=SimpleNamespace(name=tc["name"], arguments=""),
                )
            ]))
            args_json = json.dumps(tc["arguments"])
            for char in args_json:
                chunks.append(_make_stream_chunk(tool_calls=[
                    SimpleNamespace(
                        index=i,
                        id=None,
                        function=SimpleNamespace(name=None, arguments=char),
                    )
                ]))

    return _AsyncChunkIter(chunks)


def _chat_response(*, content: str, tool_calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_client(*, chat: list[Any] | None = None, responses: list[Any] | None = None) -> Any:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=_SequentialApi(chat or [])),
        responses=_SequentialApi(responses or []),
    )


def _fake_run_eval_factory(
    holder: dict[str, Any], *, reward: float, content: str
) -> Callable[..., Any]:
    @asynccontextmanager
    async def fake_run_eval(*args: Any, **kwargs: Any) -> AsyncIterator[FakeCtx]:
        _ = (args, kwargs)
        ctx = FakeCtx()
        holder["ctx"] = ctx
        yield ctx
        ctx.reward = reward
        ctx.evaluation_result = EvaluationResult(reward=reward, done=True, content=content)

    return fake_run_eval


@pytest.mark.asyncio
async def test_run_scenario_chat_interactive_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=0.9, content="done")
    )
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="lookup", arguments='{"query":"x"}'),
    )
    client = _make_client(
        chat=[
            _chat_response(content="", tool_calls=[tool_call]),
            _chat_response(content="Analysis complete."),
            _chat_response(content="Root cause identified."),
        ]
    )

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="chat_completions",
    ) as chat:
        first = await chat.send("Begin")
        second = await chat.send("Give me the root cause")
        result = await chat.finish()

    assert first.answer == "Analysis complete."
    assert second.answer == "Root cause identified."
    assert result.answer == "Root cause identified."
    assert result.reward == 0.9
    assert holder["ctx"].submitted == "Root cause identified."
    # Scenario setup prompt should be injected before first user turn.
    assert chat.messages[0]["role"] == "user"
    assert chat.messages[0]["content"] == "Solve the task"


@pytest.mark.asyncio
async def test_run_scenario_chat_interactive_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=0.4, content="ok")
    )

    client = _make_client(
        responses=[
            SimpleNamespace(
                id="resp-1",
                output_text="",
                output=[
                    SimpleNamespace(
                        type="function_call",
                        id="fc-1",
                        name="lookup",
                        arguments='{"query":"a"}',
                    )
                ],
            ),
            SimpleNamespace(id="resp-2", output_text="First response", output=[]),
        ]
    )

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="responses",
    ) as chat:
        turn = await chat.send("Analyze this")
        result = await chat.finish()

    assert turn.answer == "First response"
    assert result.answer == "First response"
    assert result.reward == 0.4


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_stream_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """send_stream yields text_delta events then turn_complete."""
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=1.0, content="ok")
    )

    client = _make_client(chat=[
        _build_stream_response(content="Hello world"),
    ])

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="chat_completions",
    ) as chat:
        events: list[ChatEvent] = []
        async for event in chat.send_stream("Hi"):
            events.append(event)
        await chat.finish()

    text_deltas = [e for e in events if e.type == "text_delta"]
    assert "".join(e.content for e in text_deltas) == "Hello world"
    assert events[-1].type == "turn_complete"
    assert events[-1].content == "Hello world"


@pytest.mark.asyncio
async def test_send_stream_with_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """send_stream yields tool_call and tool_result events during tool loop."""
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=0.8, content="done")
    )

    client = _make_client(chat=[
        _build_stream_response(
            tool_calls=[{"id": "call_1", "name": "lookup", "arguments": {"query": "x"}}]
        ),
        _build_stream_response(content="Found it"),
    ])

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="chat_completions",
    ) as chat:
        events: list[ChatEvent] = []
        async for event in chat.send_stream("Search for x"):
            events.append(event)
        await chat.finish()

    event_types = [e.type for e in events]
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "turn_complete" in event_types

    tc_event = next(e for e in events if e.type == "tool_call")
    assert tc_event.tool_name == "lookup"
    assert tc_event.tool_call_id == "call_1"

    tr_event = next(e for e in events if e.type == "tool_result")
    assert tr_event.tool_name == "lookup"
    assert tr_event.content == "ok"

    assert events[-1].content == "Found it"


# ---------------------------------------------------------------------------
# Trace header propagation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_headers_forwarded_and_merged_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=1.0, content="ok")
    )
    client = _make_client(chat=[_chat_response(content="ok")])

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="chat_completions",
        completion_kwargs={"extra_headers": {"x-custom": "1"}},
    ) as chat:
        await chat.send("Hi")
        await chat.finish()

    sent_headers = client.chat.completions.calls[0]["extra_headers"]
    assert sent_headers["Trace-Id"] == "trace-123"
    assert sent_headers["x-custom"] == "1"


@pytest.mark.asyncio
async def test_trace_headers_forwarded_and_merged_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=1.0, content="ok")
    )
    client = _make_client(chat=[_build_stream_response(content="hello")])

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="chat_completions",
        completion_kwargs={"extra_headers": {"x-custom": "1"}},
    ) as chat:
        async for _ in chat.send_stream("Hi"):
            pass
        await chat.finish()

    sent_headers = client.chat.completions.calls[0]["extra_headers"]
    assert sent_headers["Trace-Id"] == "trace-123"
    assert sent_headers["x-custom"] == "1"
    assert client.chat.completions.calls[0]["stream"] is True


@pytest.mark.asyncio
async def test_trace_headers_forwarded_and_merged_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=1.0, content="ok")
    )
    client = _make_client(
        responses=[SimpleNamespace(id="resp-1", output_text="ok", output=[])]
    )

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="responses",
        completion_kwargs={"extra_headers": {"x-custom": "1"}},
    ) as chat:
        await chat.send("Hi")
        await chat.finish()

    sent_headers = client.responses.calls[0]["extra_headers"]
    assert sent_headers["Trace-Id"] == "trace-123"
    assert sent_headers["x-custom"] == "1"


@pytest.mark.asyncio
async def test_no_user_span_when_trace_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=1.0, content="ok")
    )

    queue_span_calls = {"count": 0}

    def fake_queue_span(_span: Any) -> None:
        queue_span_calls["count"] += 1

    monkeypatch.setattr("hud.telemetry.exporter.queue_span", fake_queue_span)
    client = _make_client(chat=[_chat_response(content="ok")])

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="chat_completions",
        trace=False,
    ) as chat:
        await chat.send("Hi")
        await chat.finish()

    assert queue_span_calls["count"] == 0


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_state_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """to_state captures session state, from_state restores it."""
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=0.5, content="ok")
    )

    client = _make_client(chat=[
        _chat_response(content="First answer."),
        _chat_response(content="Second answer."),
    ])

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="chat_completions",
    ) as chat:
        await chat.send("Turn one")
        state = chat.to_state()

    assert state["model"] == "gpt-4o"
    assert state["last_answer"] == "First answer."
    assert state["trace_id"] == "trace-123"
    assert len(state["messages"]) > 0

    # Restore and verify
    ctx2 = FakeCtx()
    client2 = _make_client(chat=[
        _chat_response(content="After restore."),
    ])
    restored = ScenarioChatSession.from_state(state, client=client2, ctx=ctx2)
    assert restored.model == "gpt-4o"
    assert restored.last_answer == "First answer."
    assert restored.trace_id == "trace-123"
    assert len(restored.messages) == len(state["messages"])

    turn = await restored.send("Continue")
    assert turn.answer == "After restore."


@pytest.mark.asyncio
async def test_from_state_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restored session can call finish() to submit and get results."""
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        "hud.scenario_chat.run_eval", _fake_run_eval_factory(holder, reward=0.7, content="ok")
    )

    client = _make_client(chat=[_chat_response(content="answer")])

    async with run_scenario_chat_interactive(
        client=client,
        model="gpt-4o",
        task=SimpleNamespace(scenario="demo"),
        api="chat_completions",
    ) as chat:
        await chat.send("Go")
        state = chat.to_state()

    ctx2 = FakeCtx()
    restored = ScenarioChatSession.from_state(state, client=_make_client(), ctx=ctx2)
    result = await restored.finish("final")
    assert result.answer == "final"
    assert ctx2.submitted == "final"
