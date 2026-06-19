"""``OpenAIChatAgent`` — chat.completions ``get_response`` parsing + error path."""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

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
    return OpenAIChatAgent(OpenAIChatConfig(model="m", model_client=client))


def _response(content: str, tool_calls: list[Any]) -> Any:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        refusal=None,
        model_dump=lambda exclude_none=True: {"role": "assistant", "content": content},
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
    result = await agent.get_response(_state(agent))
    assert result.done is True
    assert result.error is not None and "boom" in result.error
