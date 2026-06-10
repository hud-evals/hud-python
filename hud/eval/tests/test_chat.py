"""``Chat`` — multi-turn conversation runner over a task."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from hud.eval import Task
from hud.eval.chat import Chat, _content_to_blocks


@pytest.fixture()
def dummy_task() -> Any:
    """Minimal Task for Chat construction."""
    return Task(env=MagicMock(), id="test_scenario")


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
    def test_requires_model(self, dummy_task: Any) -> None:
        with pytest.raises(TypeError):
            Chat(dummy_task)  # type: ignore[call-arg]

    def test_positional_task(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="test-model")
        assert chat._task is dummy_task
        assert chat._model == "test-model"

    def test_messages_start_empty(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="test-model")
        assert chat.messages == []

    def test_clear_resets_messages(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="test-model")
        chat.messages = [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        chat.clear()
        assert chat.messages == []


class TestHistory:
    def test_export_and_load_roundtrip(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m")
        chat.messages = [{"role": "user", "content": {"type": "text", "text": "hi"}}]
        exported = chat.export_history()
        assert exported == chat.messages
        assert exported is not chat.messages

        restored = Chat(dummy_task, model="m")
        restored.load_history(exported)
        assert restored.messages == exported


class TestMessageFormat:
    @pytest.mark.asyncio()
    async def test_send_stores_prompt_message_format(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="test-model")

        run = MagicMock()
        run.trace = MagicMock(content="response text", citations=[])
        fake_task = MagicMock()
        fake_task.__aenter__ = AsyncMock(return_value=run)
        fake_task.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("hud.eval.chat.replace", return_value=fake_task),
            patch.object(chat, "_create_agent", return_value=AsyncMock()),
        ):
            await chat.send("hello")

        assert len(chat.messages) == 2

        user_msg = chat.messages[0]
        assert user_msg["role"] == "user"
        assert user_msg["content"]["type"] == "text"
        assert user_msg["content"]["text"] == "hello"

        assistant_msg = chat.messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"]["type"] == "text"
        assert assistant_msg["content"]["text"] == "response text"
