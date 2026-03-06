"""A2A orchestrator executor backed by Chat with session management.

Thin A2A wrapper around :class:`Chat`.  Each A2A context gets its own
``Chat`` instance so concurrent conversations are isolated.  The
environment defines its own tools, system prompt, and routing via a
``chat=True`` scenario — the orchestrator just forwards messages.
"""

from __future__ import annotations

import asyncio
import logging
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

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

LOGGER = logging.getLogger(__name__)


class OrchestratorExecutor(AgentExecutor):
    """A2A executor backed by per-session Chat instances.

    Takes an environment name and optional scenario target, creates one
    ``Chat`` per A2A context for session isolation, and forwards turns.
    All orchestration logic lives in the environment's scenario — this
    class is pure A2A plumbing.
    """

    def __init__(
        self,
        env_name: str,
        *,
        model: str,
        scenario: str | None = "assist",
        max_steps: int = 50,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        from hud.eval.task import Task

        resolved_scenario = self._resolve_scenario(env_name, scenario)
        self._task = Task(env={"name": env_name}, scenario=resolved_scenario)
        self._model = model
        self._max_steps = max_steps
        self._name = name or f"hud-{env_name}"
        self._description = description or f"A2A agent for {env_name}"

        self._sessions: dict[str, Chat] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_last_active: dict[str, float] = {}
        self._session_ttl_seconds = 30 * 60  # 30 minutes

    @staticmethod
    def _resolve_scenario(env_name: str, scenario: str | None) -> str | None:
        """Normalize scenario input to a fully qualified identifier."""
        if scenario is None:
            return None
        if ":" in scenario:
            return scenario
        return f"{env_name}:{scenario}"

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_or_create_chat(self, context_id: str) -> Chat:
        """Return existing Chat for *context_id* or create a new one."""
        self._cleanup_stale_sessions()
        chat = self._sessions.get(context_id)
        if chat is None:
            chat = Chat(
                self._task,
                model=self._model,
                max_steps=self._max_steps,
            )
            self._sessions[context_id] = chat
        self._session_last_active[context_id] = asyncio.get_event_loop().time()
        return chat

    def _get_lock(self, context_id: str) -> asyncio.Lock:
        return self._session_locks.setdefault(context_id, asyncio.Lock())

    def _remove_session(self, context_id: str) -> None:
        session = self._sessions.pop(context_id, None)
        if session is not None:
            session.clear()
        self._session_locks.pop(context_id, None)
        self._session_last_active.pop(context_id, None)

    def _cleanup_stale_sessions(self) -> None:
        now = asyncio.get_event_loop().time()
        stale = [
            cid for cid, ts in self._session_last_active.items()
            if now - ts > self._session_ttl_seconds
        ]
        for cid in stale:
            self._remove_session(cid)
        if stale:
            LOGGER.info("Cleaned up %d stale sessions", len(stale))

    @staticmethod
    def _new_message(text: str) -> Message:
        return Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[Part(root=TextPart(text=text))],
        )

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
        status_kwargs: dict[str, Any] = {"state": state}
        if text is not None:
            status_kwargs["message"] = self._new_message(text)

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                final=final,
                status=TaskStatus(**status_kwargs),
            )
        )

    # ------------------------------------------------------------------
    # A2A agent card & serving
    # ------------------------------------------------------------------

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
        """Serve the orchestrator via the A2A Starlette app."""
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
        LOGGER.info("Serving A2A orchestrator at %s", public_url)
        uvicorn.run(app.build(), host=host, port=port)

    # ------------------------------------------------------------------
    # A2A AgentExecutor interface
    # ------------------------------------------------------------------

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or str(uuid.uuid4())
        task_id = context.task_id or str(uuid.uuid4())
        message = context.get_user_input()
        message_id = getattr(getattr(context, "message", None), "message_id", "") or ""

        await self._enqueue_status(
            event_queue,
            context_id=context_id,
            task_id=task_id,
            state=TaskState.working,
            final=False,
        )

        try:
            async with self._get_lock(context_id):
                chat = self._get_or_create_chat(context_id)
                result = await chat.send(message)
                content = result.content or ""

            LOGGER.info(
                "a2a_turn_completed context_id=%s task_id=%s "
                "message_id=%s trace_id=%s",
                context_id,
                task_id,
                message_id,
                chat.session_id,
            )

            await self._enqueue_status(
                event_queue,
                context_id=context_id,
                task_id=task_id,
                state=TaskState.input_required,
                final=True,
                text=content,
            )
        except Exception as exc:
            LOGGER.exception("orchestrator execute failed")
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
