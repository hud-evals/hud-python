"""``OpenAIAgent`` — construction + ``get_response`` parsing of the Responses API,
with a fake ``AsyncOpenAI`` client (no network).
"""
# pyright: reportPrivateUsage=false

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from openai.types.responses import ResponseOutputText

from hud.agents.openai.agent import OpenAIAgent, OpenAIRunState
from hud.agents.types import OpenAIConfig


class FakeResponses:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


class FakeOpenAI:
    def __init__(self, response: Any) -> None:
        self.responses = FakeResponses(response)


def _agent(response: Any) -> OpenAIAgent:
    return OpenAIAgent(OpenAIConfig(model="gpt-test", model_client=FakeOpenAI(response)))


def test_format_message_shapes_user_text() -> None:
    agent = _agent(SimpleNamespace(id="r", output=[]))
    msg = cast("dict[str, Any]", agent._format_message("user", "hello"))
    assert msg["role"] == "user"


def _api_response(
    id: str, output: list[Any], usage: Any = None, incomplete_details: Any = None
) -> Any:
    """A fake Responses-API payload: output items plus the response envelope."""
    return SimpleNamespace(
        id=id,
        output=output,
        model="gpt-test-v1",
        usage=usage,
        incomplete_details=incomplete_details,
    )


async def test_get_response_parses_text_and_function_call() -> None:
    response = _api_response(
        "resp_1",
        [
            SimpleNamespace(
                type="message",
                content=[ResponseOutputText(type="output_text", text="hi", annotations=[])],
            ),
            SimpleNamespace(
                type="function_call",
                name="shell",
                arguments='{"command": ["ls"]}',
                call_id="call_1",
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=9,
            output_tokens=4,
            input_tokens_details=SimpleNamespace(cached_tokens=2),
        ),
    )
    agent = _agent(response)
    state = OpenAIRunState(messages=[agent._format_message("user", "go")])

    result = await agent.get_response(state)

    assert result.content == "hi"
    assert [tc.name for tc in result.tool_calls] == ["shell"]
    assert result.tool_calls[0].arguments == {"command": ["ls"]}
    assert result.done is False
    assert state.last_response_id == "resp_1"
    # Model and usage are normalized off the provider response.
    assert result.model == "gpt-test-v1"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 9
    assert result.usage.completion_tokens == 4
    assert result.usage.cached_tokens == 2


async def test_get_response_done_when_no_tool_calls() -> None:
    response = _api_response("resp_2", [])
    agent = _agent(response)
    state = OpenAIRunState(messages=[agent._format_message("user", "hi")])

    result = await agent.get_response(state)
    assert result.done is True
    assert result.tool_calls == []
    assert result.usage is None  # provider omitted usage
    assert result.finish_reason is None


async def test_get_response_surfaces_token_cap_truncation() -> None:
    # Responses has no finish_reason; incomplete_details.reason carries the cap.
    response = _api_response(
        "resp_3", [], incomplete_details=SimpleNamespace(reason="max_output_tokens")
    )
    agent = _agent(response)
    state = OpenAIRunState(messages=[agent._format_message("user", "hi")])

    result = await agent.get_response(state)
    assert result.finish_reason == "max_output_tokens"


async def test_get_response_short_circuits_on_consumed_messages() -> None:
    agent = _agent(SimpleNamespace(id="unused", output=[]))
    state = OpenAIRunState(
        messages=[agent._format_message("user", "go")],
        last_response_id="prev",
    )
    state.message_cursor = len(state.messages)  # nothing new to send

    result = await agent.get_response(state)
    assert result.done is True
    # No API call should have been made.
    assert cast("Any", agent.openai_client.responses).calls == []


async def test_get_response_parses_shell_call() -> None:
    response = _api_response(
        "resp_3",
        [
            SimpleNamespace(
                type="shell_call",
                action=SimpleNamespace(to_dict=lambda: {"command": ["pwd"]}),
                call_id="call_sh",
            ),
        ],
    )
    agent = _agent(response)
    state = OpenAIRunState(messages=[agent._format_message("user", "run")])

    result = await agent.get_response(state)
    assert [tc.name for tc in result.tool_calls] == ["shell"]
