"""Provider outages must not masquerade as model scores in real local rollouts."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import pytest
from anthropic import AsyncAnthropic
from fastmcp import FastMCP
from fastmcp.utilities.tests import run_server_async
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from hud.agents.claude import ClaudeAgent
from hud.agents.gemini import GeminiAgent
from hud.agents.openai_compatible import OpenAIChatAgent
from hud.agents.types import AgentStep, ClaudeConfig, GeminiConfig, OpenAIChatConfig, ToolStep
from hud.capabilities import Capability
from hud.environment import Environment
from hud.eval import Job, LocalRuntime, Task
from hud.eval.run import rollout


def _completion(provider: str, answer: str) -> httpx.Response:
    if provider == "gemini":
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": answer}]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )
    if provider == "openai":
        return httpx.Response(
            200,
            json={
                "id": "completion-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": answer,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "message-test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": "",
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": answer,
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        ),
    )


def _agent(provider: str, client: httpx.AsyncClient):
    if provider == "openai":
        openai_client = AsyncOpenAI(
            api_key="test-key",
            base_url="https://provider.invalid",
            http_client=client,
            max_retries=0,
        )
        return OpenAIChatAgent(OpenAIChatConfig(model="test-model", model_client=openai_client))
    if provider == "gemini":
        gemini_client = genai.Client(
            api_key="test-key",
            http_options=genai_types.HttpOptions(
                base_url="https://provider.invalid",
                httpx_async_client=client,
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )
        return GeminiAgent(
            GeminiConfig(model="test-model", model_client=gemini_client, include_thoughts=False)
        )
    anthropic_client = AsyncAnthropic(
        api_key="test-key", base_url="https://provider.invalid", http_client=client, max_retries=0
    )
    return ClaudeAgent(ClaudeConfig(model="test-model", model_client=anthropic_client))


@pytest.fixture
def evaluation():
    env = Environment("provider_errors")
    graded: list[bool] = []
    finished: list[bool] = []

    @env.template()
    async def score(healthy: bool):
        try:
            answer = yield "healthy" if healthy else "unavailable"
            graded.append(healthy)
            yield 1.0 if answer == "pass" else 0.0
        finally:
            finished.append(healthy)

    return env, graded, finished


@pytest.mark.parametrize("provider", ["openai", "claude", "gemini"])
@pytest.mark.parametrize("failure", [429, 503, 529, "transport"])
async def test_provider_outage_is_reported_and_excluded_from_mean_reward(
    provider, failure, evaluation
):
    env, graded, finished = evaluation

    def respond(request: httpx.Request) -> httpx.Response:
        if "healthy" in request.content.decode():
            return _completion(provider, "pass")
        if failure == "transport":
            raise httpx.ReadError("provider disconnected", request=request)
        return httpx.Response(
            failure,
            json={
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "provider unavailable",
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        agent = _agent(provider, client)
        good = await rollout(
            Task(env=env.name, id="score", args={"healthy": True}), agent, runtime=LocalRuntime(env)
        )
        failed = await rollout(
            Task(env=env.name, id="score", args={"healthy": False}),
            agent,
            runtime=LocalRuntime(env),
        )

    job = Job(id="provider-errors", name="provider-errors", runs=[good, failed])
    assert job.reward == 1.0
    assert job.errors == [failed]
    assert failed.grade.is_error
    assert failed.trace.stop_reason == "provider_error"
    assert not failed.trace.is_truncated
    assert failed.trace.error
    assert failed.prompt_text == "unavailable"
    assert graded == [True]
    assert finished == [True, False]


@pytest.mark.parametrize("shared", [True, False])
async def test_provider_outage_does_not_get_overwritten_by_a_verifier(shared, evaluation):
    env, graded, finished = evaluation
    verifier_env = env if shared else Environment("verifier")
    verified: list[str] = []
    placements: list[str] = []

    @verifier_env.template()
    async def verify():
        verified.append("started")
        yield "verify"
        yield 1.0

    @asynccontextmanager
    async def runtime(task):
        placements.append(task.id)
        async with LocalRuntime(env if task.env == env.name else verifier_env)(task) as address:
            yield address

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "provider unavailable"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        run = await rollout(
            Task(
                env=env.name,
                id="score",
                args={"healthy": False},
                verifier=Task(env=verifier_env.name, id="verify"),
            ),
            _agent("openai", client),
            runtime=runtime,
        )

    assert run.grade.is_error
    assert Job(id="verifier", name="verifier", runs=[run]).errors == [run]
    assert graded == []
    assert verified == []
    assert placements == ["score"]
    assert finished == [False]


async def test_normal_wrong_answer_still_counts_as_a_zero_score(evaluation):
    env, graded, finished = evaluation
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: _completion("openai", "wrong"))
    ) as client:
        run = await rollout(
            Task(env=env.name, id="score", args={"healthy": True}),
            _agent("openai", client),
            runtime=LocalRuntime(env),
        )
    job = Job(id="wrong-answer", name="wrong-answer", runs=[run])
    assert job.reward == 0.0
    assert job.errors == []
    assert not run.grade.is_error
    assert run.grade.raw
    assert graded == [True]
    assert finished == [True]


@pytest.mark.parametrize("provider", ["openai", "claude", "gemini"])
@pytest.mark.parametrize("grader", ["guarded", "strip"])
async def test_provider_outage_is_an_error_regardless_of_grader(provider, grader):
    env = Environment("grader_errors")
    graded: list[str | None] = []
    finished: list[bool] = []

    @env.template()
    async def score():
        try:
            answer = yield "unavailable"
            graded.append(answer)
            if grader == "guarded":
                yield 1.0 if answer and answer == "pass" else 0.0
            else:
                yield 1.0 if answer.strip() == "pass" else 0.0
        finally:
            finished.append(True)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": {"message": "provider unavailable"}})
        )
    ) as client:
        run = await rollout(
            Task(env=env.name, id="score"), _agent(provider, client), runtime=LocalRuntime(env)
        )

    assert Job(id="grader-errors", name="grader-errors", runs=[run]).errors == [run]
    assert run.grade.is_error
    assert run.trace.stop_reason == "provider_error"
    assert run.prompt_text == "unavailable"
    assert graded == []
    assert finished == [True]


async def test_provider_outage_preserves_completed_turns_and_tool_results(evaluation):
    env, graded, finished = evaluation
    calls: list[str] = []
    requests = 0
    server = FastMCP("lookup")

    @server.tool()
    async def lookup() -> str:
        calls.append("lookup")
        return "found"

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests > 1:
            return httpx.Response(503, json={"error": {"message": "provider unavailable"}})
        response = _completion("openai", "Looking up the answer")
        body = response.json()
        tool = json.loads(request.content)["tools"][0]["function"]["name"]
        body["choices"][0]["finish_reason"] = "tool_calls"
        body["choices"][0]["message"]["tool_calls"] = [
            {"id": "lookup-1", "type": "function", "function": {"name": tool, "arguments": "{}"}}
        ]
        return httpx.Response(200, json=body)

    async with (
        run_server_async(server) as url,
        httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client,
    ):
        env.add_capability(Capability.mcp(name="lookup", url=url))
        run = await rollout(
            Task(env=env.name, id="score", args={"healthy": True}),
            _agent("openai", client),
            runtime=LocalRuntime(env),
        )

    assert requests == 2
    assert calls == ["lookup"]
    assert run.trace.content == "Looking up the answer"
    assert run.trace.stop_reason == "provider_error"
    assert run.grade.is_error
    assert run.trace.collect(
        lambda step: step.content if isinstance(step, AgentStep) else None
    ) == ["Looking up the answer"]
    results = run.trace.collect(lambda step: step.result if isinstance(step, ToolStep) else None)
    assert len(results) == 1
    assert not results[0].isError
    assert graded == []
    assert finished == [True]


async def test_tool_transport_error_does_not_invalidate_a_recovered_answer(evaluation):
    env, graded, finished = evaluation
    requests = 0
    server = FastMCP("lookup")

    @server.tool()
    async def lookup() -> str:
        raise httpx.ReadError("tool endpoint disconnected")

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests > 1:
            return _completion("openai", "pass")
        body = _completion("openai", "Looking up the answer").json()
        tool = json.loads(request.content)["tools"][0]["function"]["name"]
        body["choices"][0]["finish_reason"] = "tool_calls"
        body["choices"][0]["message"]["tool_calls"] = [
            {"id": "lookup-1", "type": "function", "function": {"name": tool, "arguments": "{}"}}
        ]
        return httpx.Response(200, json=body)

    async with (
        run_server_async(server) as url,
        httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client,
    ):
        env.add_capability(Capability.mcp(name="lookup", url=url))
        run = await rollout(
            Task(env=env.name, id="score", args={"healthy": True}),
            _agent("openai", client),
            runtime=LocalRuntime(env),
        )

    assert requests == 2
    assert run.trace.stop_reason == "done"
    assert run.trace.status == "completed"
    assert run.reward == 1.0
    assert not run.grade.is_error
    assert Job(id="tool-error", name="tool-error", runs=[run]).errors == []
    results = run.trace.collect(lambda step: step.result if isinstance(step, ToolStep) else None)
    assert len(results) == 1
    assert results[0].isError
    assert graded == [True]
    assert finished == [True]
