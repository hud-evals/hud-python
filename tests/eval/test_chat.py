"""Chat: a conversation folded over a chat-style task, one rollout per turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot
from mcp.types import ImageContent, TextContent

from hud import Environment
from hud.agents.base import Agent
from hud.agents.types import AgentStep, Citation
from hud.eval import Chat, LocalRuntime, Task

if TYPE_CHECKING:
    from hud.eval import Run


def assistant_env(prompts: list[Any]) -> Environment:
    """A chat-style environment: its prompt is the whole conversation so far."""
    env = Environment("chat")

    @env.template()
    async def assistant(messages: list[dict[str, Any]]):
        prompts.append(messages)
        yield messages
        yield 1.0

    return env


class Echo(Agent):
    """Echoes the last user turn; fails on ``fail`` and cites its source on ``cite``."""

    async def __call__(self, run: Run) -> None:
        last = run.prompt_messages[-1].content
        text = last.text if isinstance(last, TextContent) else "<non-text>"
        if text == "fail":
            raise RuntimeError("model unavailable")
        citations = [Citation(type="url_citation", text=text, source="https://docs.test")]
        run.record(AgentStep(content=f"echo:{text}", citations=citations if text == "cite" else []))
        run.trace.content = f"echo:{text}"


async def test_a_conversation_keeps_its_history_and_one_job_across_turns() -> None:
    prompts: list[Any] = []
    chat = Chat(
        Task(env="chat", id="assistant"), Echo(), runtime=LocalRuntime(assistant_env(prompts))
    )
    blocks = [
        TextContent(type="text", text="what is in this picture?"),
        ImageContent(type="image", data="aW1n", mimeType="image/png"),
    ]

    replies = [(await chat.send(turn)).content for turn in ("hello", blocks, "cite")]
    with pytest.raises(RuntimeError, match=r"\[agent loop\] RuntimeError: model unavailable"):
        await chat.send("fail")

    assert replies == ["echo:hello", "echo:<non-text>", "echo:cite"]
    assert chat.messages == snapshot(
        [
            {
                "role": "user",
                "content": {"type": "text", "text": "hello", "annotations": None, "meta": None},
            },
            {"role": "assistant", "content": {"type": "text", "text": "echo:hello"}},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "what is in this picture?",
                        "annotations": None,
                        "meta": None,
                    },
                    {
                        "type": "image",
                        "data": "aW1n",
                        "mimeType": "image/png",
                        "annotations": None,
                        "meta": None,
                    },
                ],
            },
            {"role": "assistant", "content": {"type": "text", "text": "echo:<non-text>"}},
            {
                "role": "user",
                "content": {"type": "text", "text": "cite", "annotations": None, "meta": None},
            },
            {
                "role": "assistant",
                "content": {"type": "text", "text": "echo:cite"},
                "citations": [
                    {"type": "url_citation", "text": "cite", "source": "https://docs.test"}
                ],
            },
            {
                "role": "user",
                "content": {"type": "text", "text": "fail", "annotations": None, "meta": None},
            },
        ]
    )
    assert prompts[-1] == chat.messages
    assert chat.job is not None
    assert [run.job_id for run in chat.job.runs] == [chat.job.id] * 4
    assert chat.job.name == "assistant"


async def test_a_chat_without_a_runtime_says_how_to_place_it() -> None:
    chat = Chat(Task(env="chat", id="assistant"), Echo())

    with pytest.raises(RuntimeError, match=r"Chat needs a runtime to converse against"):
        await chat.send("hello")

    assert (chat.messages, chat.job) == (
        [
            {
                "role": "user",
                "content": {"type": "text", "text": "hello", "annotations": None, "meta": None},
            }
        ],
        None,
    )
