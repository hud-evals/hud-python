"""``ClaudeAgent`` — ``get_response`` parsing over a fake streaming Messages client,
plus the pure ``_citation`` / ``_cache_last_user_block`` helpers.
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from hud.agents.claude.agent import ClaudeAgent


class FakeStream:
    def __init__(self, final: Any) -> None:
        self._final = final

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *_a: Any) -> bool:
        return False

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_message(self) -> Any:
        return self._final


class FakeMessages:
    def __init__(self, final: Any) -> None:
        self._final = final

    def stream(self, **_kwargs: Any) -> FakeStream:
        return FakeStream(self._final)


class FakeAnthropic:
    def __init__(self, final: Any) -> None:
        self.beta = SimpleNamespace(messages=FakeMessages(final))


def _agent(final: Any) -> ClaudeAgent:
    from hud.agents.types import ClaudeConfig

    return ClaudeAgent(
        ClaudeConfig(model="claude-test", max_tokens=1024, model_client=FakeAnthropic(final))
    )


def _state(agent: ClaudeAgent) -> Any:
    from hud.agents.tool_agent import RunState

    return RunState(messages=[agent._format_message("user", "go")])


def test_format_message_shape() -> None:
    agent = _agent(SimpleNamespace(content=[], stop_reason="end_turn"))
    msg = agent._format_message("assistant", "hi")
    assert msg["role"] == "assistant"


async def test_get_response_text_and_tool_use() -> None:
    final = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="hello", citations=None),
            SimpleNamespace(type="tool_use", id="t1", name="bash", input={"command": "ls"}),
        ],
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


async def test_get_response_done_on_text_only() -> None:
    final = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="done", citations=None)],
        stop_reason="end_turn",
    )
    agent = _agent(final)
    result = await agent.get_response(_state(agent))
    assert result.done is True
    assert result.content == "done"
    assert result.tool_calls == []


async def test_get_response_collects_thinking() -> None:
    final = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="pondering"),
            SimpleNamespace(type="text", text="answer", citations=None),
        ],
        stop_reason="end_turn",
    )
    agent = _agent(final)
    result = await agent.get_response(_state(agent))
    assert result.reasoning == "pondering"


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
