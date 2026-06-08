"""``GeminiAgent`` — ``get_response`` parsing over a fake Generate Content client,
plus ``_make_tool_call`` mapping and ``_grounding_citations``.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from hud.agents.gemini.agent import GeminiAgent, _grounding_citations
from hud.agents.tool_agent import RunState
from hud.agents.types import GeminiConfig


class FakeModels:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def generate_content(self, **_kwargs: Any) -> Any:
        return self._response


class FakeGenai:
    def __init__(self, response: Any) -> None:
        self.aio = SimpleNamespace(models=FakeModels(response))


def _agent(response: Any) -> GeminiAgent:
    agent = GeminiAgent.__new__(GeminiAgent)
    a = cast("Any", agent)
    a.config = GeminiConfig(model="gemini-test", include_thoughts=False)
    a.gemini_client = FakeGenai(response)
    a.max_recent_turn_with_screenshots = 3
    return agent


def _state(agent: GeminiAgent) -> Any:
    from hud.agents.tool_agent import RunState

    return RunState[Any, Any](messages=[agent._format_message("user", "go")])


def test_format_message_uses_model_role() -> None:
    agent = _agent(SimpleNamespace(candidates=[]))
    assert agent._format_message("assistant", "hi").role == "model"
    assert agent._format_message("user", "hi").role == "user"


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
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=resp_content,
                grounding_metadata=None,
                finish_reason=SimpleNamespace(name="STOP"),
            )
        ]
    )
    agent = _agent(response)

    result = await agent.get_response(_state(agent))

    assert result.content == "hi"
    assert [tc.name for tc in result.tool_calls] == ["bash"]
    assert result.done is False
    assert result.finish_reason == "STOP"


async def test_get_response_done_text_only() -> None:
    resp_content = SimpleNamespace(
        role="model",
        parts=[SimpleNamespace(function_call=None, text="answer", thought=None)],
    )
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=resp_content, grounding_metadata=None, finish_reason=None)
        ]
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
    state = RunState[Any, Any](
        tools=cast("Any", {"computer_use": SimpleNamespace(provider_name="computer_use")}),
    )
    tc = agent._make_tool_call(cast("Any", fc), state)
    assert tc.name == "computer_use"
    assert tc.arguments == {"action": "click_at", "x": 1}
    assert tc.provider_name == "click_at"


def test_make_tool_call_plain_function() -> None:
    agent = _agent(SimpleNamespace(candidates=[]))
    fc = cast("Any", SimpleNamespace(name="bash", args={"command": "ls"}))
    tc = agent._make_tool_call(fc, RunState())
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
