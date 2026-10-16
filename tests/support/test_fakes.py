"""Scripted model replies parse in each provider's real SDK, so the fake keeps their wire format."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessageParam, BetaToolUseBlock
from google import genai
from google.genai.types import HttpOptions
from openai import AsyncOpenAI
from openai.types.responses import ResponseFunctionToolCall

from tests.support import call, say

if TYPE_CHECKING:
    from tests.support import Models


async def test_a_scripted_anthropic_turn_streams_through_the_anthropic_sdk(models: Models) -> None:
    models.script([call("bash", command="ls"), say("done", reasoning="thinking it over")])
    client = AsyncAnthropic(base_url=models.url, api_key="k")
    messages: list[BetaMessageParam] = [{"role": "user", "content": "go"}]

    async with client.beta.messages.stream(model="m", max_tokens=64, messages=messages) as stream:
        first = await stream.get_final_message()
    (tool_use,) = first.content
    assert isinstance(tool_use, BetaToolUseBlock)
    messages += [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use.id,
                    "name": tool_use.name,
                    "input": tool_use.input,
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "a.txt"}],
        },
    ]
    async with client.beta.messages.stream(model="m", max_tokens=64, messages=messages) as stream:
        second = await stream.get_final_message()

    assert first.stop_reason == "tool_use"
    assert (tool_use.name, tool_use.input) == ("bash", {"command": "ls"})
    assert [block.type for block in second.content] == ["thinking", "text"]
    assert models.requests()[1].tool_results == ["a.txt"]


async def test_scripted_responses_chain_turns_by_previous_response_id(models: Models) -> None:
    models.script([call("lookup", q="x"), say("found it")])
    client = AsyncOpenAI(base_url=models.url, api_key="k")

    first = await client.responses.create(model="m", input="find x")
    (function_call,) = first.output
    assert isinstance(function_call, ResponseFunctionToolCall)
    second = await client.responses.create(
        model="m",
        previous_response_id=first.id,
        input=[{"type": "function_call_output", "call_id": function_call.call_id, "output": "x=1"}],
    )

    assert function_call.name == "lookup"
    assert second.output_text == "found it"
    assert models.requests()[1].tool_results == ["x=1"]


async def test_a_scripted_gemini_turn_parses_in_the_genai_sdk(models: Models) -> None:
    models.script([call("click_at", x=10, y=20)])
    client = genai.Client(
        api_key="k", http_options=HttpOptions(base_url=models.url, api_version="v1beta")
    )

    response = await client.aio.models.generate_content(model="gemini-test", contents="click it")

    (function_call,) = response.function_calls or []
    assert (function_call.name, function_call.args) == ("click_at", {"x": 10, "y": 20})
    assert models.requests()[0].model == "gemini-test"
