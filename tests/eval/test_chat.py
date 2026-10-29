"""``Chat``: a conversation folded over a chat-style task, one rollout per turn."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from mcp.types import ImageContent, TextContent

from hud import Environment
from hud.agents.base import Agent
from hud.eval import Chat, LocalRuntime, Task
from tests.conftest import eventually
from tests.fixtures.services import Reply

if TYPE_CHECKING:
    from hud.eval import Run
    from tests.fixtures.services import FakeServices


def assistant(prompts: list[Any]) -> Environment:
    """A chat-style environment: its prompt is the whole conversation so far."""
    env = Environment("chat")

    @env.template()
    async def reply(messages: list[dict[str, Any]]):
        prompts.append(messages)
        yield messages
        yield 1.0

    return env


class Echo(Agent):
    """Replies ``echo:`` and the last turn's text, and fails on ``fail``."""

    async def __call__(self, run: Run) -> None:
        last = run.prompt_messages[-1].content
        text = last.text if isinstance(last, TextContent) else "<image>"
        if text == "fail":
            raise RuntimeError("model unavailable")
        run.trace.content = f"echo:{text}"


async def test_each_turn_runs_on_the_whole_conversation_and_adds_the_reply_to_it() -> None:
    prompts: list[Any] = []
    chat = Chat(Task(env="chat", id="reply"), Echo(), runtime=LocalRuntime(assistant(prompts)))
    picture = [
        TextContent(type="text", text="what is this?"),
        ImageContent(type="image", data="aW1n", mimeType="image/png"),
    ]

    first = await chat.send("hello")
    second = await chat.send(picture)

    assert (first.content, second.content) == ("echo:hello", "echo:<image>")
    assert [message["role"] for message in chat.messages] == ["user", "assistant"] * 2
    texts = [m["content"]["text"] for m in chat.messages if isinstance(m["content"], dict)]
    assert texts == ["hello", "echo:hello", "echo:<image>"]
    assert [block["type"] for block in chat.messages[2]["content"]] == ["text", "image"]
    assert prompts == [chat.messages[:1], chat.messages[:3]]


async def test_a_failed_turn_raises_and_leaves_no_reply_in_the_conversation() -> None:
    chat = Chat(Task(env="chat", id="reply"), Echo(), runtime=LocalRuntime(assistant([])))

    with pytest.raises(RuntimeError, match="model unavailable"):
        await chat.send("fail")

    assert [message["role"] for message in chat.messages] == ["user"]


class HeldEcho(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, run: Run) -> None:
        content = run.prompt_messages[-1].content
        assert isinstance(content, TextContent)
        if content.text == "first":
            self.started.set()
            await self.release.wait()
        run.trace.content = f"echo:{content.text}"


async def test_concurrent_sends_keep_the_conversation_and_job_in_order(
    reporting: FakeServices,
) -> None:
    prompts: list[Any] = []
    agent = HeldEcho()
    runtime = LocalRuntime(lambda task: assistant(prompts))
    chat = Chat(Task(env="chat", id="reply"), agent, runtime=runtime)
    job_enter = "/v2/trace/job/{id}/enter"
    reporting.route("api", "POST", job_enter, Reply(json={}, delay=0.3))
    first = asyncio.create_task(chat.send("first"))
    await eventually(lambda: len(reporting.requests("api", "POST", job_enter)) == 1)
    second = asyncio.create_task(chat.send("second"))
    await agent.started.wait()
    try:
        await asyncio.wait({second}, timeout=0.3)
    finally:
        agent.release.set()
    first_trace, second_trace = await asyncio.gather(first, second)

    assert (first_trace.content, second_trace.content) == ("echo:first", "echo:second")
    assert [(m["role"], m["content"]["text"]) for m in chat.messages] == [
        ("user", "first"),
        ("assistant", "echo:first"),
        ("user", "second"),
        ("assistant", "echo:second"),
    ]
    assert prompts == [chat.messages[:1], chat.messages[:3]]
    assert chat.job is not None
    assert len(chat.job.runs) == 2
    assert {run.job_id for run in chat.job.runs} == {chat.job.id}
    assert len(reporting.requests("api", "POST", job_enter)) == 1


async def test_cancelling_a_waiting_send_does_not_add_a_turn() -> None:
    agent = HeldEcho()
    chat = Chat(
        Task(env="chat", id="reply"), agent, runtime=LocalRuntime(lambda task: assistant([]))
    )
    first = asyncio.create_task(chat.send("first"))
    await agent.started.wait()
    waiting = asyncio.create_task(chat.send("second"))
    await asyncio.sleep(0)
    waiting.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await waiting
    finally:
        agent.release.set()
    await first

    assert [(m["role"], m["content"]["text"]) for m in chat.messages] == [
        ("user", "first"),
        ("assistant", "echo:first"),
    ]


async def test_a_failed_turn_releases_the_next_send() -> None:
    class FailsFirst(HeldEcho):
        async def __call__(self, run: Run) -> None:
            await super().__call__(run)
            if run.trace.content == "echo:first":
                raise RuntimeError("model unavailable")

    agent = FailsFirst()
    chat = Chat(
        Task(env="chat", id="reply"), agent, runtime=LocalRuntime(lambda task: assistant([]))
    )
    first = asyncio.create_task(chat.send("first"))
    await agent.started.wait()
    second = asyncio.create_task(chat.send("second"))
    agent.release.set()
    with pytest.raises(RuntimeError, match="model unavailable"):
        await first
    assert (await second).content == "echo:second"
    assert [(m["role"], m["content"]["text"]) for m in chat.messages] == [
        ("user", "first"),
        ("user", "second"),
        ("assistant", "echo:second"),
    ]


async def test_separate_chats_can_progress_while_a_turn_is_held() -> None:
    agent = HeldEcho()
    runtime = LocalRuntime(lambda task: assistant([]))
    held = Chat(Task(env="chat", id="reply"), agent, runtime=runtime)
    other = Chat(Task(env="chat", id="reply"), agent, runtime=runtime)
    first = asyncio.create_task(held.send("first"))
    await agent.started.wait()
    try:
        response = await asyncio.wait_for(other.send("second"), timeout=5)
        assert response.content == "echo:second"
    finally:
        agent.release.set()
        await first
