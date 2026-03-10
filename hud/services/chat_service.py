"""A2A chat service backed by per-session Chat instances."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import AgentExecutor
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from hud.services.chat import Chat
from hud.services.reply_metadata import build_reply_metadata_event

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

    from hud.eval.task import Task

LOGGER = logging.getLogger(__name__)


class ChatService(AgentExecutor):
    """Thin A2A wrapper around per-session ``Chat`` instances."""

    def __init__(
        self,
        task: Task,
        /,
        *,
        model: str,
        max_steps: int = 50,
        name: str | None = None,
        description: str | None = None,
        trace: bool = True,
        quiet: bool = True,
    ) -> None:
        self._task = task
        self._model = model
        self._max_steps = max_steps
        self._name = name or task.scenario or "chat-service"
        self._description = description or f"A2A service for {task.scenario or 'tasks'}"
        self._trace = trace
        self._quiet = quiet

        self._sessions: dict[str, Chat] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_last_active: dict[str, float] = {}
        self._session_ttl_seconds = 30 * 60

    def _get_or_create_chat(self, context_id: str) -> Chat:
        self._cleanup_stale_sessions()
        chat = self._sessions.get(context_id)
        if chat is None:
            chat = Chat(
                self._task,
                model=self._model,
                max_steps=self._max_steps,
                trace=self._trace,
                quiet=self._quiet,
            )
            self._sessions[context_id] = chat
        self._session_last_active[context_id] = time.monotonic()
        return chat

    def _remove_session(self, context_id: str) -> None:
        session = self._sessions.pop(context_id, None)
        if session is not None:
            session.clear()
        lock = self._session_locks.get(context_id)
        # Preserve an in-flight lock so concurrent requests for the same
        # context cannot create a second lock and run in parallel.
        if lock is None or not lock.locked():
            self._session_locks.pop(context_id, None)
        self._session_last_active.pop(context_id, None)

    def _cleanup_stale_sessions(self) -> None:
        now = time.monotonic()
        stale = [
            cid
            for cid, ts in self._session_last_active.items()
            if now - ts > self._session_ttl_seconds
        ]
        for cid in stale:
            self._remove_session(cid)
        if stale:
            LOGGER.info("Cleaned up %d stale sessions", len(stale))

    # ------------------------------------------------------------------
    # Direct Python usage (session-based)
    # ------------------------------------------------------------------

    async def send(
        self,
        message: str,
        *,
        session_id: str = "default",
    ) -> Any:
        """Send a message to a session and get the agent's response.

        Each session_id gets an independent conversation with its own history.
        Use this for multi-user scenarios (e.g. a web app with per-user chats).

        Args:
            message: The user message text.
            session_id: Identifies the conversation. Different IDs get
                independent Chat instances with separate history.

        Returns:
            Trace with the agent's response in ``trace.content``.
        """
        async with self._session_locks.setdefault(session_id, asyncio.Lock()):
            chat = self._get_or_create_chat(session_id)
            return await chat.send(message)

    def clear(self, session_id: str = "default") -> None:
        """Clear a session's conversation history."""
        self._remove_session(session_id)

    def export_history(self, session_id: str = "default") -> list[dict[str, Any]]:
        """Export a session's conversation history for persistence."""
        chat = self._sessions.get(session_id)
        if chat is None:
            return []
        return chat.export_history()

    def load_history(
        self, messages: list[dict[str, Any]], session_id: str = "default"
    ) -> None:
        """Restore conversation history into a session."""
        chat = self._get_or_create_chat(session_id)
        chat.load_history(messages)

    # ------------------------------------------------------------------
    # A2A internals
    # ------------------------------------------------------------------

    async def _enqueue_status(
        self,
        event_queue: EventQueue,
        *,
        context_id: str,
        task_id: str,
        state: TaskState,
        final: bool,
        text: str | None = None,
    ) -> None:
        status = TaskStatus(state=state)
        if text is not None:
            status = TaskStatus(
                state=state,
                message=Message(
                    message_id=str(uuid.uuid4()),
                    role=Role.agent,
                    parts=[Part(root=TextPart(text=text))],
                ),
            )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                final=final,
                status=status,
            )
        )

    def agent_card(self, url: str = "http://localhost:9999/") -> AgentCard:
        return AgentCard(
            name=self._name,
            description=self._description,
            url=url,
            version="1.0",
            capabilities=AgentCapabilities(streaming=True),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[],
        )

    def serve(
        self,
        *,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 9999,
        url: str | None = None,
    ) -> None:
        """Serve the chat service via the A2A Starlette app."""
        import uvicorn
        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore

        public_url = url or f"http://{host}:{port}/"
        handler = DefaultRequestHandler(
            agent_executor=self,
            task_store=InMemoryTaskStore(),
        )
        app = A2AStarletteApplication(
            agent_card=self.agent_card(public_url),
            http_handler=handler,
        )
        LOGGER.info("Serving A2A chat service at %s", public_url)
        uvicorn.run(app.build(), host=host, port=port)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or str(uuid.uuid4())
        task_id = context.task_id or str(uuid.uuid4())
        message = context.get_user_input()

        await self._enqueue_status(
            event_queue,
            context_id=context_id,
            task_id=task_id,
            state=TaskState.working,
            final=False,
        )

        try:
            async with self._session_locks.setdefault(context_id, asyncio.Lock()):
                chat = self._get_or_create_chat(context_id)
                result = await chat.send(message)
                content = result.content or ""


            metadata_event = build_reply_metadata_event(
                context_id=context_id,
                task_id=task_id,
                trace=result,
            )
            if metadata_event is not None:
                await event_queue.enqueue_event(metadata_event)

            await self._enqueue_status(
                event_queue,
                context_id=context_id,
                task_id=task_id,
                state=TaskState.input_required,
                final=True,
                text=content,
            )
        except Exception as exc:
            LOGGER.exception("chat service execute failed")
            await self._enqueue_status(
                event_queue,
                context_id=context_id,
                task_id=task_id,
                state=TaskState.failed,
                final=True,
                text=str(exc),
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or ""
        task_id = context.task_id or ""

        self._remove_session(context_id)

        await self._enqueue_status(
            event_queue,
            context_id=context_id,
            task_id=task_id,
            state=TaskState.canceled,
            final=True,
        )
