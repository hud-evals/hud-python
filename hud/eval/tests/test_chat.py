"""``Chat`` — multi-turn conversation runner over a task.

Turn tests place each turn's rollout with ``runtime=SubprocessRuntime(env_file)`` — a pure-data
``Task`` row against a chat-style env served from a child process.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import pytest
from mcp.types import TextContent

from hud.agents.base import Agent
from hud.eval import SubprocessRuntime, Task
from hud.eval.chat import Chat, _content_to_blocks

if TYPE_CHECKING:
    from pathlib import Path


class _EchoAgent(Agent):
    """Replies with ``echo:<last user message>`` read from the prompt."""

    async def __call__(self, run: Any) -> None:
        last = run.prompt[-1]["content"]["text"]
        run.trace.content = f"echo:{last}"


@pytest.fixture()
def dummy_task() -> Any:
    """Minimal Task for Chat construction."""
    return Task(env="chat", id="test_scenario")


class TestContentHelpers:
    def test_content_to_blocks_string(self) -> None:
        blocks = _content_to_blocks("hello")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == "hello"

    def test_content_to_blocks_passthrough(self) -> None:
        original = [TextContent(type="text", text="x")]
        assert _content_to_blocks(original) is original


class TestChatConstruction:
    def test_requires_an_agent(self, dummy_task: Any) -> None:
        with pytest.raises(TypeError):
            Chat(dummy_task)  # type: ignore[call-arg]

    def test_messages_start_empty_and_are_the_public_history(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, _EchoAgent())
        assert chat.messages == []
        assert chat.job is None  # the conversation's job starts on the first send
        # History is the plain ``messages`` list: persist/restore it directly.
        chat.messages = [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        assert Chat(dummy_task, _EchoAgent()).messages == []


_CHAT_ENV = """\
from hud import Environment

env = Environment("chat")


@env.template()
async def assistant(messages: list):
    _answer = yield messages
    yield 1.0
"""


@pytest.fixture(scope="module")
def chat_env_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("chat") / "env.py"
    path.write_text(textwrap.dedent(_CHAT_ENV), encoding="utf-8")
    return path


def _chat_task() -> Task:
    """A pure data row for the chat-style task the spawned file defines."""
    return Task(env="chat", id="assistant", args={"messages": []})


class TestSend:
    async def test_send_runs_a_turn_and_stores_prompt_message_format(
        self, chat_env_file: Path
    ) -> None:
        chat = Chat(_chat_task(), _EchoAgent(), runtime=SubprocessRuntime(chat_env_file))

        trace = await chat.send("hello")

        assert trace.content == "echo:hello"
        assert len(chat.messages) == 2

        user_msg = chat.messages[0]
        assert user_msg["role"] == "user"
        assert user_msg["content"]["type"] == "text"
        assert user_msg["content"]["text"] == "hello"

        assistant_msg = chat.messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"]["type"] == "text"
        assert assistant_msg["content"]["text"] == "echo:hello"

    async def test_one_job_spans_the_conversation(self, chat_env_file: Path) -> None:
        chat = Chat(_chat_task(), _EchoAgent(), runtime=SubprocessRuntime(chat_env_file))

        await chat.send("hello")
        await chat.send("again")

        job = chat.job
        assert job is not None
        assert len(job.runs) == 2
        # Every turn's trace reports under the conversation's job.
        assert {run.job_id for run in job.runs} == {job.id}

    async def test_failed_turn_raises_and_records_no_assistant_message(
        self, chat_env_file: Path
    ) -> None:
        class _Boom(Agent):
            async def __call__(self, run: Any) -> None:
                raise RuntimeError("agent exploded")

        chat = Chat(_chat_task(), _Boom(), runtime=SubprocessRuntime(chat_env_file))

        with pytest.raises(RuntimeError, match="agent exploded"):
            await chat.send("hello")

        assert [m["role"] for m in chat.messages] == ["user"]
