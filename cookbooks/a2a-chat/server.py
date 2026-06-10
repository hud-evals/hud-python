"""Serve a HUD chat task over the A2A protocol.

A2A (and any other wire protocol) is a frontend over :class:`hud.Chat`: the
executor below translates A2A requests into ``chat.send()`` calls, keeping an
independent ``Chat`` (and so an independent conversation) per A2A context.

This is reference code, not part of the SDK — copy and adapt it. See the
README in this directory for setup and usage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    Artifact,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from hud import Chat, Environment, Runtime, spawn
from hud.agents import create_agent
from hud.eval import Task

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

    from hud.agents.base import Agent
    from hud.environment import Provider
    from hud.types import Trace

LOGGER = logging.getLogger("a2a_chat_server")

SESSION_TTL_SECONDS = 30 * 60


def _status_event(
    context_id: str, task_id: str, state: TaskState, *, final: bool, text: str | None = None
) -> TaskStatusUpdateEvent:
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
    return TaskStatusUpdateEvent(context_id=context_id, task_id=task_id, final=final, status=status)


def _citations_event(context_id: str, task_id: str, trace: Trace) -> TaskArtifactUpdateEvent | None:
    """Transport reply citations as a structured artifact, if any."""
    if not trace.citations:
        return None
    payload = {"type": "hud_reply_metadata", "citations": trace.citations, "data": None}
    return TaskArtifactUpdateEvent(
        context_id=context_id,
        task_id=task_id,
        append=False,
        last_chunk=True,
        artifact=Artifact(
            artifact_id=str(uuid.uuid4()),
            name="hud_reply_metadata",
            parts=[Part(root=TextPart(text=json.dumps(payload)))],
        ),
    )


class ChatExecutor(AgentExecutor):
    """A2A adapter: one ``Chat`` (conversation) per A2A context id."""

    def __init__(self, task: Task, agent: Agent, *, on: Provider | None = None) -> None:
        self._task = task
        self._agent = agent
        self._on = on
        self._sessions: dict[str, Chat] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_active: dict[str, float] = {}

    def _chat(self, context_id: str) -> Chat:
        now = time.monotonic()
        for cid, ts in list(self._last_active.items()):
            if now - ts > SESSION_TTL_SECONDS:
                self._sessions.pop(cid, None)
                self._last_active.pop(cid, None)
                lock = self._locks.get(cid)
                if lock is None or not lock.locked():
                    self._locks.pop(cid, None)
        chat = self._sessions.setdefault(context_id, Chat(self._task, self._agent, on=self._on))
        self._last_active[context_id] = now
        return chat

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or str(uuid.uuid4())
        task_id = context.task_id or str(uuid.uuid4())
        message = context.get_user_input()

        await event_queue.enqueue_event(
            _status_event(context_id, task_id, TaskState.working, final=False)
        )
        try:
            async with self._locks.setdefault(context_id, asyncio.Lock()):
                result = await self._chat(context_id).send(message)

            citations = _citations_event(context_id, task_id, result)
            if citations is not None:
                await event_queue.enqueue_event(citations)
            await event_queue.enqueue_event(
                _status_event(
                    context_id,
                    task_id,
                    TaskState.input_required,
                    final=True,
                    text=result.content or "",
                )
            )
        except Exception as exc:
            LOGGER.exception("chat execute failed")
            await event_queue.enqueue_event(
                _status_event(context_id, task_id, TaskState.failed, final=True, text=str(exc))
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        context_id = context.context_id or ""
        self._sessions.pop(context_id, None)
        self._last_active.pop(context_id, None)
        await event_queue.enqueue_event(
            _status_event(context_id, context.task_id or "", TaskState.canceled, final=True)
        )


def serve(task: Task, agent: Agent, *, on: Provider | None, host: str, port: int) -> None:
    name = task.id or "chat"
    url = f"http://{host}:{port}/"
    app = A2AStarletteApplication(
        agent_card=AgentCard(
            name=name,
            description=f"A2A service for {name}",
            url=url,
            version="1.0",
            capabilities=AgentCapabilities(streaming=True),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=[],
        ),
        http_handler=DefaultRequestHandler(
            agent_executor=ChatExecutor(task, agent, on=on),
            task_store=InMemoryTaskStore(),
        ),
    )
    LOGGER.info("Serving A2A chat at %s", url)
    uvicorn.run(app.build(), host=host, port=port)


def main() -> None:
    """Serve `HUD_TASK` (default: this directory's chat_env.py) over A2A.

    Placement: `HUD_ENV_URL` attaches each turn to an already-served control
    channel; otherwise every turn spawns `HUD_SOURCE` locally.
    """
    task_id = os.getenv("HUD_TASK", "chat_full").strip()
    env_name = os.getenv("HUD_ENV", "chat").strip()
    env_url = os.getenv("HUD_ENV_URL", "").strip()
    source = os.getenv("HUD_SOURCE", str(Path(__file__).parent / "chat_env.py")).strip()
    placement = Runtime(env_url) if env_url else spawn(source)

    serve(
        Task(env=Environment(env_name), id=task_id),
        create_agent(
            os.getenv("HUD_MODEL", "claude-haiku-4-5"),
            max_steps=int(os.getenv("HUD_MAX_STEPS", "50")),
        ),
        on=placement,
        host=os.getenv("HUD_A2A_HOST", "0.0.0.0"),  # noqa: S104
        port=int(os.getenv("HUD_A2A_PORT", "9999")),
    )


if __name__ == "__main__":
    main()
