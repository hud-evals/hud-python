"""``OpenAIAgent`` — construction + ``get_response`` parsing of the Responses API,
with a fake ``AsyncOpenAI`` client (no network).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import mcp.types as mcp_types
import pytest
from openai.types.responses import ResponseOutputText

from hud.agents.openai.agent import EmptyShellCallError, OpenAIAgent, OpenAIRunState
from hud.agents.openai.tools.base import format_openai_result
from hud.agents.openai.tools.computer import OPENAI_COMPUTER_SPEC, OpenAIComputerTool
from hud.agents.types import OpenAIConfig
from hud.capabilities import Capability, RFBClient
from hud.clients.client import HudClient
from hud.eval.run import Run
from hud.types import MCPToolCall, MCPToolResult


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


def test_format_openai_result_empty_output_emits_one_text_item() -> None:
    # The Responses API rejects a function_call_output with an empty output
    # list, so a contentless tool result must still produce one input_text item.
    call = MCPToolCall(id="call_1", name="noop")
    formatted = cast("dict[str, Any]", format_openai_result(call, MCPToolResult(content=[])))

    assert formatted["type"] == "function_call_output"
    assert formatted["call_id"] == "call_1"
    assert formatted["output"] == [{"type": "input_text", "text": ""}]


def test_format_openai_result_sends_structured_content_only_without_content() -> None:
    call = MCPToolCall(id="call_1", name="lookup")
    mirrored = MCPToolResult(
        content=[mcp_types.TextContent(type="text", text='{"answer": 42}')],
        structuredContent={"answer": 42},
    )
    structured_only = MCPToolResult(content=[], structuredContent={"answer": 42})

    with_content = cast("dict[str, Any]", format_openai_result(call, mirrored))
    without_content = cast("dict[str, Any]", format_openai_result(call, structured_only))

    assert with_content["output"] == [{"type": "input_text", "text": '{"answer": 42}'}]
    assert without_content["output"] == [{"type": "input_text", "text": '{"answer": 42}'}]


def test_format_computer_result_preserves_screenshot_mime_type() -> None:
    agent = _agent(SimpleNamespace(id="r", output=[]))
    tool = OpenAIComputerTool(spec=OPENAI_COMPUTER_SPEC, client=cast("Any", object()))
    state = OpenAIRunState(tools={"computer": tool})
    result = MCPToolResult(
        content=[
            mcp_types.ImageContent(
                type="image",
                data="d2VicA==",
                mimeType="image/webp",
            ),
        ],
    )

    formatted = cast(
        "dict[str, Any]",
        agent._format_result(MCPToolCall(id="call_1", name="computer"), result, state),
    )

    assert formatted["output"]["image_url"] == "data:image/webp;base64,d2VicA=="


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


@pytest.fixture
def computer_client() -> Mock:
    screen = Mock(
        spec=RFBClient,
        drain=AsyncMock(),
        screenshot_png=AsyncMock(return_value=(b"screen", "image/png")),
    )
    return Mock(
        spec=HudClient,
        manifest=SimpleNamespace(
            bindings=[Capability.rfb(name="screen", url="rfb://localhost:5900")],
        ),
        open=AsyncMock(return_value=screen),
        start_task=AsyncMock(return_value={"prompt": "Use the computer."}),
        grade=AsyncMock(return_value={"score": 1.0}),
    )


@pytest.mark.parametrize(
    ("actions", "error"),
    [
        ([{"type": "screenshot"}, {"type": "response", "text": "all done"}], None),
        ([], "actions list is empty"),
        ([{"type": "keypress", "keys": ["UNKNOWN"]}, {"type": "type", "text": "later"}], "UNKNOWN"),
    ],
    ids=["response", "empty-call", "failed-keypress"],
)
async def test_computer_result_reaches_next_model_turn(
    computer_client: Mock, actions: list[dict[str, Any]], error: str | None
) -> None:
    screen = computer_client.open.return_value
    screen.conn.keyboard.press.side_effect = KeyError("UNKNOWN")
    call = SimpleNamespace(
        type="computer_call",
        call_id="call_1",
        actions=[SimpleNamespace(to_dict=lambda action=action: action) for action in actions],
        action=None,
        pending_safety_checks=[],
    )
    create = AsyncMock(side_effect=[_api_response("resp_1", [call]), _api_response("resp_2", [])])
    agent = OpenAIAgent(
        OpenAIConfig(model="gpt-test", model_client=Mock(responses=Mock(create=create))),
    )

    async with Run(computer_client, "computer-task", {}) as run:
        await agent(run)

    assert run.trace.status == "completed"
    assert create.await_count == 2
    feedback = create.call_args_list[1].kwargs["input"]
    assert feedback[0]["type"] == "computer_call_output"
    assert feedback[0]["call_id"] == "call_1"
    assert feedback[0]["output"]["image_url"].startswith("data:image/png;base64,")
    if error:
        assert error in feedback[1]["content"][0]["text"]
        assert (
            "Remaining actions in this call were not executed" in feedback[1]["content"][0]["text"]
        )
    screen.conn.keyboard.write.assert_not_called()


async def test_computer_error_preserves_failed_screenshot_context(computer_client: Mock) -> None:
    computer_client.open.return_value.screenshot_png.side_effect = RuntimeError(
        "display unavailable"
    )
    call = SimpleNamespace(
        type="computer_call",
        call_id="call_1",
        actions=[],
        action=SimpleNamespace(to_dict=lambda: {"type": "frobnicate"}),
        pending_safety_checks=[],
    )
    agent = _agent(_api_response("resp_1", [call]))

    async with Run(computer_client, "computer-task", {}) as run:
        await agent(run)

    assert run.trace.status == "error"
    error = next(step.error for step in run.trace.steps if step.error)
    assert "frobnicate" in error
    assert "display unavailable" in error


async def test_computer_screenshot_timeout_preserves_action_context(computer_client: Mock) -> None:
    timeout = TimeoutError("capture timed out")
    screen = computer_client.open.return_value
    screen.screenshot_png.side_effect = timeout
    tool = OpenAIComputerTool(spec=OPENAI_COMPUTER_SPEC, client=screen)

    with pytest.raises(TimeoutError, match="capture timed out") as raised:
        await tool.execute({"type": "frobnicate"})

    assert raised.value is timeout
    assert any("frobnicate" in note for note in raised.value.__notes__)


@pytest.mark.parametrize("error_type", [TimeoutError, asyncio.CancelledError])
async def test_computer_action_interruption_stops_batch(
    computer_client: Mock, error_type: type[BaseException]
) -> None:
    screen = computer_client.open.return_value
    error = error_type("input interrupted")
    screen.drain = AsyncMock(side_effect=error)
    tool = OpenAIComputerTool(spec=OPENAI_COMPUTER_SPEC, client=screen)

    with pytest.raises(error_type) as raised:
        await tool.execute(
            {"actions": [{"type": "move", "x": 3, "y": 4}, {"type": "type", "text": "later"}]},
        )

    assert raised.value is error
    screen.conn.keyboard.write.assert_not_called()
    screen.screenshot_png.assert_not_awaited()


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


async def test_get_response_reuses_prompt_cache_key() -> None:
    agent = _agent(_api_response("resp_cache", []))
    state = OpenAIRunState(messages=[agent._format_message("user", "first")])

    await agent.get_response(state)
    state.messages.append(agent._format_message("user", "second"))
    await agent.get_response(state)

    calls = cast("Any", agent.openai_client.responses).calls
    assert calls[0]["prompt_cache_key"] == agent.config.prompt_cache_key
    assert calls[1]["prompt_cache_key"] == agent.config.prompt_cache_key


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


async def test_get_response_rejects_empty_shell_call() -> None:
    # A shell_call with an empty commands array poisons the stored response
    # chain (OpenAI 500s every continuation via previous_response_id), so the
    # turn must fail fast and its response id must never be committed.
    response = _api_response(
        "resp_poisoned",
        [
            SimpleNamespace(
                type="shell_call",
                action=SimpleNamespace(to_dict=lambda: {"commands": [], "timeout_ms": 10000}),
                call_id="call_empty",
            ),
        ],
    )
    agent = _agent(response)
    state = OpenAIRunState(messages=[agent._format_message("user", "run")])

    with pytest.raises(EmptyShellCallError, match="empty commands array"):
        await agent.get_response(state)

    assert state.last_response_id is None
