"""``OpenAIChatAgent`` — chat.completions ``get_response`` parsing + error path."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from hud.agents.openai_compatible.agent import OpenAIChatAgent, OpenAIChatRunState
from hud.agents.types import OpenAIChatConfig


class FakeCompletions:
    def __init__(self, response: Any, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    async def create(self, **_kwargs: Any) -> Any:
        if self._error is not None:
            raise self._error
        return self._response


class FakeOpenAI:
    def __init__(self, response: Any, error: Exception | None = None) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(response, error))


def _agent(response: Any, error: Exception | None = None) -> OpenAIChatAgent:
    client = cast("Any", FakeOpenAI(response, error))
    return OpenAIChatAgent(OpenAIChatConfig(model="m", openai_client=client))


def _response(content: str, tool_calls: list[Any]) -> Any:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        refusal=None,
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": content},
    )
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    return SimpleNamespace(choices=[choice])


def _state(agent: OpenAIChatAgent) -> Any:
    return OpenAIChatRunState(messages=[agent._format_message("user", "go")])


async def test_get_response_text_only() -> None:
    agent = _agent(_response("hi", []))
    result = await agent.get_response(_state(agent))
    assert result.content == "hi"
    assert result.done is True
    assert result.tool_calls == []


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


async def test_get_response_propagates_api_errors() -> None:
    # API failures must surface (and end the rollout via the loop's handler), not be
    # silently turned into a successful-looking terminal AgentResponse.
    agent = _agent(None, error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await agent.get_response(_state(agent))


async def test_get_response_raises_on_empty_choices() -> None:
    agent = _agent(SimpleNamespace(choices=[]))
    with pytest.raises(ValueError, match="no choices"):
        await agent.get_response(_state(agent))
