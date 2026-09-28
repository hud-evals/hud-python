"""``GeminiAgent`` — ``get_response`` parsing over a fake Generate Content client,
plus ``_make_tool_call`` mapping and ``_grounding_citations``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import mcp.types as mcp_types

from hud.agents.gemini.agent import GeminiAgent, _grounding_citations
from hud.agents.tool_agent import RunState
from hud.agents.types import GeminiConfig
from hud.types import MCPToolResult, Step, Trace

if TYPE_CHECKING:
    from hud.eval.run import Run


class FakeModels:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def generate_content(self, **_kwargs: Any) -> Any:
        return self._response


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
    agent = _agent(SimpleNamespace(candidates=[]))
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


class RecordingModels:
    """Replays scripted responses, recording the contents each request carried."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.requests: list[Any] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        # Snapshot: the agent goes on mutating the list it passed.
        self.requests.append(list(kwargs["contents"]))
        return self._responses.pop(0)


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
