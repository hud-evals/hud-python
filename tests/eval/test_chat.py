"""``Chat``: a conversation folded over a chat-style task, one rollout per turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from mcp.types import ImageContent, TextContent

from hud import Environment
from hud.agents.base import Agent
from hud.agents.types import AgentStep, Citation
from hud.eval import Chat, LocalRuntime, Task

if TYPE_CHECKING:
    from hud.eval import Run


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


async def test_every_turn_reports_under_the_conversations_one_job() -> None:
    chat = Chat(Task(env="chat", id="reply"), Echo(), runtime=LocalRuntime(assistant([])))

    await chat.send("hello")
    await chat.send("again")

    assert chat.job is not None
    assert (chat.job.name, [run.job_id for run in chat.job.runs]) == ("reply", [chat.job.id] * 2)


async def test_a_reply_keeps_the_citations_its_agent_gave() -> None:
    class Cites(Agent):
        async def __call__(self, run: Run) -> None:
            citation = Citation(type="url_citation", text="docs", source="https://docs.test")
            run.record(AgentStep(content="see the docs", citations=[citation]))
            run.trace.content = "see the docs"

    chat = Chat(Task(env="chat", id="reply"), Cites(), runtime=LocalRuntime(assistant([])))

    await chat.send("where?")

    assert chat.messages[-1]["citations"] == [
        {"type": "url_citation", "text": "docs", "source": "https://docs.test"}
    ]


async def test_a_failed_turn_raises_and_leaves_no_reply_in_the_conversation() -> None:
    chat = Chat(Task(env="chat", id="reply"), Echo(), runtime=LocalRuntime(assistant([])))

    with pytest.raises(RuntimeError, match="model unavailable"):
        await chat.send("fail")

    assert [message["role"] for message in chat.messages] == ["user"]
