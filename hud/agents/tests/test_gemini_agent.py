"""``GeminiAgent`` — ``get_response`` over a fake streaming Generate Content client,
plus ``_make_tool_call`` mapping and ``_grounding_citations``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import mcp.types as mcp_types
from google import genai
from google.genai import types as genai_types

from hud.agents.gemini.agent import GeminiAgent, _grounding_citations
from hud.agents.tool_agent import RunState
from hud.agents.types import GeminiConfig
from hud.types import MCPToolResult, Step, Trace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval.run import Run


async def _chunks(*chunks: Any) -> AsyncIterator[Any]:
    for chunk in chunks:
        yield chunk


class FakeModels:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def generate_content_stream(self, **_kwargs: Any) -> AsyncIterator[Any]:
        return _chunks(self._response)


class FakeGenai:
    def __init__(self, response: Any) -> None:
        self.aio = SimpleNamespace(models=FakeModels(response))


def _agent(response: Any) -> GeminiAgent:
    return GeminiAgent(
        GeminiConfig(model="gemini-test", include_thoughts=False, model_client=FakeGenai(response))
    )


def _state(agent: GeminiAgent) -> Any:
    return RunState(messages=[agent._format_message("user", "go")])


def test_format_message_uses_model_role() -> None:
    agent = _agent(SimpleNamespace(candidates=[]))
    assert agent._format_message("assistant", "hi").role == "model"
    assert agent._format_message("user", "hi").role == "user"


def _api_response(*candidates: Any) -> Any:
    """A fake ``GenerateContentResponse``: candidates plus the response envelope."""
    return SimpleNamespace(
        candidates=list(candidates),
        model_version="gemini-test-v2",
        usage_metadata=SimpleNamespace(
            prompt_token_count=5,
            candidates_token_count=3,
            cached_content_token_count=None,
        ),
    )


async def test_get_response_text_and_function_call() -> None:
    resp_content = SimpleNamespace(
        role="model",
        parts=[
            SimpleNamespace(function_call=None, text="hi", thought=None),
            SimpleNamespace(
                function_call=SimpleNamespace(name="bash", args={"command": "ls"}),
                text=None,
                thought=None,
            ),
        ],
    )
    response = _api_response(
        SimpleNamespace(
            content=resp_content,
            grounding_metadata=None,
            finish_reason=SimpleNamespace(name="STOP"),
        )
    )
    agent = _agent(response)

    result = await agent.get_response(_state(agent))

    assert result.content == "hi"
    assert [tc.name for tc in result.tool_calls] == ["bash"]
    assert result.done is False
    assert result.finish_reason == "STOP"
    # Model and usage are normalized off the provider response.
    assert result.model == "gemini-test-v2"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 3


async def test_get_response_done_text_only() -> None:
    resp_content = SimpleNamespace(
        role="model",
        parts=[SimpleNamespace(function_call=None, text="answer", thought=None)],
    )
    response = _api_response(
        SimpleNamespace(content=resp_content, grounding_metadata=None, finish_reason=None)
    )
    agent = _agent(response)
    result = await agent.get_response(_state(agent))
    assert result.done is True
    assert result.content == "answer"


async def test_get_response_no_candidates_raises() -> None:
    agent = _agent(SimpleNamespace(candidates=[], model_version=None, usage_metadata=None))
    try:
        await agent.get_response(_state(agent))
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError for empty candidates")


def test_make_tool_call_maps_predefined_to_computer() -> None:
    agent = _agent(SimpleNamespace(candidates=[]))
    fc = SimpleNamespace(name="click_at", args={"x": 1})
    tc = agent._make_tool_call(cast("Any", fc), cast("Any", SimpleNamespace(name="computer_use")))
    assert tc.name == "computer_use"
    assert tc.arguments == {"action": "click_at", "x": 1}
    assert tc.provider_name == "click_at"


def test_make_tool_call_plain_function() -> None:
    agent = _agent(SimpleNamespace(candidates=[]))
    fc = cast("Any", SimpleNamespace(name="bash", args={"command": "ls"}))
    tc = agent._make_tool_call(fc, None)
    assert tc.name == "bash"
    assert tc.arguments == {"command": "ls"}


def test_grounding_citations() -> None:
    meta = SimpleNamespace(
        grounding_chunks=[SimpleNamespace(web=SimpleNamespace(uri="http://x", title="T"))],
        grounding_supports=[
            SimpleNamespace(
                segment=SimpleNamespace(text="seg", start_index=0, end_index=3),
                grounding_chunk_indices=[0],
            )
        ],
    )
    cites = _grounding_citations(cast("Any", meta))
    assert len(cites) == 1
    assert cites[0].source == "http://x"
    assert cites[0].type == "grounding"


def _chunk(*parts: genai_types.Part, finish_reason: str | None = None) -> Any:
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=list(parts)),
                finish_reason=genai_types.FinishReason(finish_reason) if finish_reason else None,
            )
        ],
        model_version="gemini-test-v2",
    )


async def test_streamed_chunks_form_one_model_turn() -> None:
    """Every streamed part joins one model turn as received; text and thoughts concatenate."""
    parts = [
        genai_types.Part(text="Checking ", thought=True),
        genai_types.Part(text="divisors.", thought=True),
        genai_types.Part(text="It is 17 *"),
        genai_types.Part(text=" 23."),
        genai_types.Part(
            function_call=genai_types.FunctionCall(name="bash", args={"command": "ls"}),
            thought_signature=b"sig-1",
        ),
        genai_types.Part(text="", thought_signature=b"sig-2"),
    ]
    stream = _chunks(
        _chunk(*parts[:2]),
        _chunk(*parts[2:4]),
        _chunk(*parts[4:], finish_reason="STOP"),
    )

    async def generate_content_stream(**_kwargs: Any) -> AsyncIterator[Any]:
        return stream

    agent = GeminiAgent(
        GeminiConfig(
            model="gemini-test",
            model_client=SimpleNamespace(
                aio=SimpleNamespace(
                    models=SimpleNamespace(generate_content_stream=generate_content_stream)
                )
            ),
        )
    )
    state = _state(agent)

    result = await agent.get_response(state)

    assert result.content == "It is 17 * 23."
    assert result.reasoning == "Checking divisors."
    assert [tc.name for tc in result.tool_calls] == ["bash"]
    assert result.finish_reason == "STOP"
    assert result.model == "gemini-test-v2"
    turn = state.messages[-1]
    assert turn.role == "model"
    assert turn.parts == parts


def _sse_client(*hops: list[dict[str, Any]]) -> tuple[genai.Client, list[httpx.Request]]:
    """A ``genai.Client`` whose n-th request streams the n-th hop's events over SSE."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        events = hops[len(requests) - 1]
        return httpx.Response(
            200,
            content=b"".join(
                b"data: " + json.dumps(event).encode() + b"\r\n\r\n" for event in events
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = genai.Client(
        api_key="hud-key",
        http_options=genai_types.HttpOptions(
            base_url="https://gateway.test",
            api_version="v1beta",
            async_client_args={"transport": httpx.MockTransport(handler)},
        ),
    )
    return client, requests


async def test_turn_streams_through_the_sdk() -> None:
    """A turn arrives over the SDK's SSE stream and is recorded as one model turn."""
    events: list[dict[str, Any]] = [
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "Thinking.", "thought": True}]}}
            ]
        },
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "39"}]}}]},
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "1"}, {"text": "", "thoughtSignature": "c2ln"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 2,
                "thoughtsTokenCount": 5,
                "totalTokenCount": 14,
            },
            "modelVersion": "gemini-4-argon",
        },
    ]
    client, requests = _sse_client(events)
    agent = GeminiAgent(GeminiConfig(model="gemini-4-argon", model_client=client))
    state = _state(agent)

    result = await agent.get_response(state)

    [request] = requests
    assert request.url.path == "/v1beta/models/gemini-4-argon:streamGenerateContent"
    assert request.url.params["alt"] == "sse"
    assert result.content == "391"
    assert result.reasoning == "Thinking."
    assert result.finish_reason == "STOP"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 7
    turn = state.messages[-1]
    assert [(part.text, part.thought, part.thought_signature) for part in turn.parts or []] == [
        ("Thinking.", True, None),
        ("39", None, None),
        ("1", None, None),
        ("", None, b"sig"),
    ]


async def test_turn_past_the_per_request_cap_continues_as_one_turn() -> None:
    """A stream that stops at the per-request cap resumes with its continuation token, and
    both hops form one model turn."""
    first_hop: list[dict[str, Any]] = [
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Working it out.", "thought": True}],
                    }
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "", "thoughtSignature": "c2lnLTE="}],
                    },
                    "finishReason": "CONTINUATION",
                    "continuationToken": "dG9rZW4tMQ==",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "thoughtsTokenCount": 65526,
                "totalTokenCount": 65536,
            },
        },
    ]
    second_hop: list[dict[str, Any]] = [
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "42"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2,
                "totalTokenCount": 12,
            },
            "modelVersion": "gemini-4-argon",
        }
    ]
    client, requests = _sse_client(first_hop, second_hop)
    agent = GeminiAgent(GeminiConfig(model="gemini-4-argon", model_client=client))
    state = _state(agent)

    result = await agent.get_response(state)

    first, second = (json.loads(request.content) for request in requests)
    assert second == {**first, "continuationToken": "dG9rZW4tMQ=="}
    assert result.content == "42"
    assert result.reasoning == "Working it out."
    assert result.done is True
    assert result.finish_reason == "STOP"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 20
    turns = [content for content in state.messages if content.role == "model"]
    assert len(turns) == 1
    assert [(part.text, part.thought_signature) for part in turns[0].parts or []] == [
        ("Working it out.", None),
        ("", b"sig-1"),
        ("42", None),
    ]


async def test_malformed_function_call_is_a_normalized_error() -> None:
    response = _api_response(
        SimpleNamespace(
            content=None,
            grounding_metadata=None,
            finish_reason=SimpleNamespace(name="MALFORMED_FUNCTION_CALL"),
        )
    )
    agent = _agent(response)
    result = await agent.get_response(_state(agent))
    assert result.error == "Provider returned a malformed function call"
    assert result.stop_reason == "malformed_tool_call"
    assert result.finish_reason == "MALFORMED_FUNCTION_CALL"
    assert result.usage is not None


class ConfigRecordingModels:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.configs: list[Any] = []

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.configs.append(kwargs["config"])
        return _chunks(self._response)


async def _sent_config(**config: Any) -> Any:
    resp_content = SimpleNamespace(
        role="model",
        parts=[SimpleNamespace(function_call=None, text="answer", thought=None)],
    )
    models = ConfigRecordingModels(
        _api_response(
            SimpleNamespace(content=resp_content, grounding_metadata=None, finish_reason=None)
        )
    )
    agent = GeminiAgent(
        GeminiConfig(
            model="gemini-test",
            include_thoughts=False,
            model_client=SimpleNamespace(aio=SimpleNamespace(models=models)),
            **config,
        )
    )
    await agent.get_response(_state(agent))
    return models.configs[0]


async def test_output_cap_defaults_to_the_model_limit() -> None:
    assert (await _sent_config()).max_output_tokens is None


async def test_configured_output_cap_is_sent() -> None:
    assert (await _sent_config(max_output_tokens=4096)).max_output_tokens == 4096


async def test_sampling_defaults_to_the_model() -> None:
    sent = await _sent_config()
    assert (sent.temperature, sent.top_p, sent.top_k) == (None, None, None)


async def test_configured_sampling_is_sent() -> None:
    sent = await _sent_config(temperature=0.5, top_p=0.9, top_k=20)
    assert (sent.temperature, sent.top_p, sent.top_k) == (0.5, 0.9, 20)


class RecordingModels:
    """Replays scripted responses, recording the contents each request carried."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []

    async def generate_content_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        # Snapshot: the agent goes on mutating the list it passed.
        self.requests.append(list(kwargs["contents"]))
        return _chunks(self._responses.pop(0))


class RecordingGenai:
    def __init__(self, responses: list[Any]) -> None:
        self.models = RecordingModels(responses)
        self.aio = SimpleNamespace(models=self.models)


class BigOutputTool:
    """Tool stub whose text block exceeds the configured budget."""

    name = "big"
    provider_name = "big"

    def __init__(self, text: str) -> None:
        self._text = text

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        del arguments
        return MCPToolResult(content=[mcp_types.TextContent(type="text", text=self._text)])


class FakeRun:
    """Offline stand-in for ``Run``: records steps onto a local trace only."""

    def __init__(self) -> None:
        self.trace = Trace()

    def record(self, step: Step) -> None:
        self.trace.record(step)


def _model_turn(*parts: Any) -> Any:
    return _api_response(
        SimpleNamespace(
            content=SimpleNamespace(role="model", parts=list(parts)),
            grounding_metadata=None,
            finish_reason=None,
        )
    )


def _tool_output(contents: Any) -> str:
    parts = contents[-1].parts
    assert parts is not None
    function_response = parts[0].function_response
    assert function_response is not None
    response = function_response.response
    assert response is not None
    return cast("str", response["output"])


async def _run_with_tool_output(text: str, budget: int) -> list[Any]:
    """Drive one tool call through the loop, returning the requests Gemini saw."""
    client = RecordingGenai(
        [
            _model_turn(
                SimpleNamespace(
                    function_call=SimpleNamespace(name="big", args={}),
                    text=None,
                    thought=None,
                )
            ),
            _model_turn(SimpleNamespace(function_call=None, text="done", thought=None)),
        ]
    )
    agent = GeminiAgent(
        GeminiConfig(
            model="gemini-test",
            include_thoughts=False,
            model_client=client,
            max_tool_result_chars=budget,
            max_steps=3,
        )
    )
    state = _state(agent)
    state.tools = {"big": cast("Any", BigOutputTool(text))}

    await agent._loop(cast("Run", FakeRun()), state)

    assert len(client.models.requests) == 2
    return client.models.requests


async def test_oversized_tool_output_is_bounded_before_the_next_request() -> None:
    requests = await _run_with_tool_output("h" * 500 + "t" * 500, budget=100)

    output = _tool_output(requests[1])
    assert "900 omitted" in output
    assert "[900 characters omitted]" in output
    assert "h" * 20 in output
    assert "h" * 21 not in output
    assert "t" * 80 in output


async def test_tool_output_within_budget_reaches_the_model_intact() -> None:
    requests = await _run_with_tool_output("short", budget=100)

    assert _tool_output(requests[1]) == "short"
