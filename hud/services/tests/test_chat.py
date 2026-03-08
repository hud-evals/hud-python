"""Tests for Chat -- multi-turn conversation wrapper and A2A executor."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import TaskArtifactUpdateEvent, TaskState, TaskStatusUpdateEvent
from mcp.types import TextContent

from hud.services.chat import Chat, _content_to_blocks

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dummy_task() -> Any:
    """Minimal Task-like object for Chat construction."""
    task = MagicMock()
    task.scenario = "test_scenario"
    task.args = {}
    task.model_copy = MagicMock(return_value=task)
    task.env = MagicMock()
    return task


# ---------------------------------------------------------------------------
# Unit tests: content helpers
# ---------------------------------------------------------------------------


class TestContentHelpers:
    def test_content_to_blocks_string(self) -> None:
        blocks = _content_to_blocks("hello")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == "hello"

    def test_content_to_blocks_passthrough(self) -> None:
        original = [TextContent(type="text", text="x")]
        assert _content_to_blocks(original) is original


# ---------------------------------------------------------------------------
# Chat construction
# ---------------------------------------------------------------------------


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

    def test_session_id_stable(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="test-model")
        assert chat._session_id  # not empty
        sid = chat._session_id
        assert chat._session_id == sid  # same across accesses

    def test_name_from_scenario(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m", name="Custom Agent")
        assert chat._name == "Custom Agent"

    def test_name_default_from_task(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m")
        assert chat._name == "test_scenario"


# ---------------------------------------------------------------------------
# Message format (PromptMessage-compatible)
# ---------------------------------------------------------------------------


class TestMessageFormat:
    @pytest.mark.asyncio()
    async def test_send_stores_prompt_message_format(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="test-model")

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "response text"
        mock_result.citations = []
        mock_agent.run = AsyncMock(return_value=mock_result)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(chat, "_create_agent", return_value=mock_agent),
            patch("hud.eval.manager.run_eval") as mock_eval,
        ):
            mock_eval.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_eval.return_value.__aexit__ = AsyncMock(return_value=False)

            # Patch the import inside send()
            import hud
            original_eval = hud.eval
            hud.eval = MagicMock(return_value=mock_ctx)
            try:
                await chat.send("hello")
            finally:
                hud.eval = original_eval

        assert len(chat.messages) == 2

        user_msg = chat.messages[0]
        assert user_msg["role"] == "user"
        assert user_msg["content"]["type"] == "text"
        assert user_msg["content"]["text"] == "hello"

        assistant_msg = chat.messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"]["type"] == "text"
        assert assistant_msg["content"]["text"] == "response text"


# ---------------------------------------------------------------------------
# A2A AgentExecutor interface
# ---------------------------------------------------------------------------


class TestA2AExecutor:
    def test_is_agent_executor(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m")
        assert isinstance(chat, AgentExecutor)

    @pytest.mark.asyncio()
    async def test_execute_enqueues_completed(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m")

        with patch.object(chat, "send", new_callable=AsyncMock) as mock_send:
            mock_result = MagicMock()
            mock_result.content = "done"
            mock_result.citations = []
            mock_send.return_value = mock_result

            context = MagicMock(spec=RequestContext)
            context.context_id = "ctx-1"
            context.task_id = "task-1"
            context.get_user_input.return_value = "hello"

            queue = EventQueue()

            await chat.execute(context, queue)

            event = await queue.dequeue_event(no_wait=True)
            assert isinstance(event, TaskStatusUpdateEvent)
            assert event.status.state == TaskState.working

            event2 = await queue.dequeue_event(no_wait=True)
            assert isinstance(event2, TaskStatusUpdateEvent)
            assert event2.status.state == TaskState.input_required
            assert event2.final is True

    @pytest.mark.asyncio()
    async def test_execute_enqueues_metadata_artifact_before_final_status(
        self, dummy_task: Any
    ) -> None:
        chat = Chat(dummy_task, model="m")

        with patch.object(chat, "send", new_callable=AsyncMock) as mock_send:
            mock_result = MagicMock()
            mock_result.content = "done"
            mock_result.citations = [
                {"type": "url_citation", "source": "https://example.com", "title": "Example"}
            ]
            mock_send.return_value = mock_result

            context = MagicMock(spec=RequestContext)
            context.context_id = "ctx-1"
            context.task_id = "task-1"
            context.get_user_input.return_value = "hello"

            queue = EventQueue()

            await chat.execute(context, queue)

            event = await queue.dequeue_event(no_wait=True)
            assert isinstance(event, TaskStatusUpdateEvent)
            assert event.status.state == TaskState.working

            event2 = await queue.dequeue_event(no_wait=True)
            assert isinstance(event2, TaskArtifactUpdateEvent)
            payload = json.loads(cast(Any, event2.artifact.parts[0].root).text)
            assert payload["type"] == "hud_reply_metadata"
            assert payload["citations"][0]["source"] == "https://example.com"
            assert payload["data"] is None

            event3 = await queue.dequeue_event(no_wait=True)
            assert isinstance(event3, TaskStatusUpdateEvent)
            assert event3.status.state == TaskState.input_required
            assert event3.final is True

    @pytest.mark.asyncio()
    async def test_execute_enqueues_failed_on_error(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m")

        with patch.object(chat, "send", side_effect=ValueError("boom")):
            context = MagicMock(spec=RequestContext)
            context.context_id = "ctx-1"
            context.task_id = "task-1"
            context.get_user_input.return_value = "hello"

            queue = EventQueue()

            await chat.execute(context, queue)

            # Should have working + failed events
            events = []
            while True:
                try:
                    events.append(await queue.dequeue_event(no_wait=True))
                except asyncio.QueueEmpty:
                    break

            states = [e.status.state for e in events]
            assert TaskState.failed in states

    @pytest.mark.asyncio()
    async def test_cancel_clears_messages(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m")
        chat.messages = [{"role": "user", "content": {"type": "text", "text": "hi"}}]

        context = MagicMock(spec=RequestContext)
        context.context_id = "ctx-1"
        context.task_id = "task-1"
        queue = EventQueue()

        await chat.cancel(context, queue)
        assert chat.messages == []


# ---------------------------------------------------------------------------
# AgentCard generation
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_agent_card_has_skill(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m", name="TestBot", description="A test bot")
        card = chat.agent_card(url="http://localhost:8000/")
        assert card.name == "TestBot"
        assert card.description == "A test bot"
        assert len(card.skills) == 1
        assert card.skills[0].id == "test_scenario"

    def test_agent_card_default_modes(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m")
        card = chat.agent_card()
        assert "text/plain" in card.default_input_modes
        assert "text/plain" in card.default_output_modes


# ---------------------------------------------------------------------------
# as_tool
# ---------------------------------------------------------------------------


class TestAsToolIntegration:
    def test_as_tool_returns_agent_tool(self, dummy_task: Any) -> None:
        chat = Chat(dummy_task, model="m")
        tool = chat.as_tool(name="my_tool")
        assert tool.name == "my_tool"
