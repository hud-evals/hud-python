"""Multi-session chat management for A2A orchestrators."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hud.services.chat import Chat, MessageContent
from hud.services.types import ChatDefinition, SessionState

if TYPE_CHECKING:
    from hud.types import Trace

LOGGER = logging.getLogger(__name__)


class ChatManagerError(RuntimeError):
    """Base class for chat manager runtime errors."""


class UnknownChatDefinitionError(ChatManagerError):
    """Raised when a requested chat definition does not exist."""


class SessionNotFoundError(ChatManagerError):
    """Raised when a session id cannot be found."""


class SessionBusyError(ChatManagerError):
    """Raised when a session already has an in-flight request."""


class SessionExpiredError(ChatManagerError):
    """Raised when a session was expired by TTL cleanup."""


class SessionFinishedError(ChatManagerError):
    """Raised when a session was explicitly finished."""


@dataclass(slots=True)
class ManagedSession:
    """State for one managed chat session."""

    session_id: str
    definition_name: str
    chat: Chat
    state: SessionState = SessionState.ACTIVE
    created_at: float = field(default_factory=time.monotonic)
    last_active_at: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_active_at = time.monotonic()


class ChatManager:
    """Session-aware manager over `Chat` instances.

    MVP protocol contract:
    - canonical key is `session_id` (mapped from A2A `context_id`)
    - one in-flight `send()` per session
    - explicit session states with deterministic terminal behavior
    """

    def __init__(
        self,
        definitions: list[ChatDefinition],
        *,
        default_definition: str | None = None,
        session_ttl_seconds: int = 30 * 60,
    ) -> None:
        if not definitions:
            raise ValueError("ChatManager requires at least one ChatDefinition")
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be >= 1")

        self._definitions = {d.name: d for d in definitions}
        if default_definition is None:
            self._default_definition = definitions[0].name
        elif default_definition in self._definitions:
            self._default_definition = default_definition
        else:
            raise UnknownChatDefinitionError(f"Unknown default definition: {default_definition}")

        self._session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, ManagedSession] = {}
        self._expired_sessions: set[str] = set()
        self._finished_sessions: set[str] = set()
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def default_definition(self) -> str:
        return self._default_definition

    def start(self) -> bool:
        """Start periodic TTL cleanup loop.

        Returns True when the cleanup task is started; False when no running
        event loop is available yet (e.g. called before server loop starts).
        """
        if self._cleanup_task is None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return False
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        return True

    async def close(self) -> None:
        """Stop cleanup loop and clear all active sessions."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

        for sid in list(self._sessions):
            self._force_close_session(sid, SessionState.FINISHED)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            self.cleanup_expired()

    def cleanup_expired(self) -> list[str]:
        """Expire sessions that have been idle longer than TTL."""
        now = time.monotonic()
        expired: list[str] = []
        for sid, session in list(self._sessions.items()):
            if session.state is not SessionState.ACTIVE:
                continue
            idle_seconds = now - session.last_active_at
            if idle_seconds > self._session_ttl_seconds:
                self._force_close_session(sid, SessionState.EXPIRED)
                expired.append(sid)
        return expired

    def list_definitions(self) -> list[str]:
        return sorted(self._definitions.keys())

    def get_session_state(self, session_id: str) -> SessionState:
        if session_id in self._expired_sessions:
            return SessionState.EXPIRED
        if session_id in self._finished_sessions:
            return SessionState.FINISHED
        if session := self._sessions.get(session_id):
            return session.state
        raise SessionNotFoundError(f"Session not found: {session_id}")

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return active in-memory session metadata."""
        return [
            {
                "session_id": s.session_id,
                "definition_name": s.definition_name,
                "state": s.state.value,
                "idle_seconds": int(time.monotonic() - s.last_active_at),
            }
            for s in self._sessions.values()
        ]

    def has_session(self, session_id: str) -> bool:
        """Return whether an active in-memory session exists."""
        return session_id in self._sessions

    def get_definition(self, name: str | None = None) -> ChatDefinition:
        """Return a chat definition by name (or default when omitted)."""
        return self._resolve_definition(name)

    def create_session(
        self,
        *,
        definition_name: str | None = None,
        session_id: str | None = None,
        task_args: dict[str, Any] | None = None,
    ) -> str:
        """Create a new active session and return its id."""
        self.cleanup_expired()
        definition = self._resolve_definition(definition_name)
        sid = session_id or uuid.uuid4().hex
        if sid in self._sessions:
            return sid
        if sid in self._expired_sessions or sid in self._finished_sessions:
            raise ChatManagerError(f"Session id cannot be reused: {sid}")

        task = definition.task
        if task_args:
            task = task.model_copy(update={"args": {**(task.args or {}), **task_args}})

        self._sessions[sid] = ManagedSession(
            session_id=sid,
            definition_name=definition.name,
            chat=Chat(
                task,
                model=definition.model,
                agent_params=definition.agent_params,
                name=definition.display_name,
                description=definition.description,
            ),
        )
        return sid

    async def send(
        self,
        session_id: str,
        message: MessageContent,
        *,
        definition_name: str | None = None,
        task_args: dict[str, Any] | None = None,
    ) -> Trace:
        """Send one turn to the session's chat.

        If session does not exist yet, it is created with `definition_name`
        (or default definition when omitted).
        """
        self.cleanup_expired()
        if session_id not in self._sessions:
            self._raise_if_terminal(session_id)
            self.create_session(
                definition_name=definition_name,
                session_id=session_id,
                task_args=task_args,
            )
        session = self._get_active_session_or_raise(session_id)

        if session.lock.locked():
            raise SessionBusyError(f"Session already has in-flight turn: {session_id}")

        async with session.lock:
            session.touch()
            chat_name = session.chat.agent_card().name
            LOGGER.info(
                "chat_send_start session_id=%s chat_name=%s trace_id=%s",
                session_id,
                chat_name,
                session.chat.session_id,
            )
            result = await session.chat.send(message)
            session.touch()
            LOGGER.info(
                "chat_send_end session_id=%s chat_name=%s trace_id=%s error=%s",
                session_id,
                chat_name,
                session.chat.session_id,
                result.isError,
            )
            return result

    def finish(self, session_id: str) -> None:
        """Finish and clear a session."""
        session = self._get_active_session_or_raise(session_id)
        if session.lock.locked():
            raise SessionBusyError(f"Cannot finish in-flight session: {session_id}")
        session.state = SessionState.FINISHING
        self._force_close_session(session_id, SessionState.FINISHED)

    def clear_session(self, session_id: str) -> None:
        """Alias for finish() for API symmetry."""
        self.finish(session_id)

    def _resolve_definition(self, name: str | None) -> ChatDefinition:
        definition_name = name or self._default_definition
        try:
            return self._definitions[definition_name]
        except KeyError as e:
            raise UnknownChatDefinitionError(f"Unknown chat definition: {definition_name}") from e

    def _raise_if_terminal(self, session_id: str) -> None:
        if session_id in self._expired_sessions:
            raise SessionExpiredError(f"Session expired: {session_id}")
        if session_id in self._finished_sessions:
            raise SessionFinishedError(f"Session finished: {session_id}")

    def _get_active_session_or_raise(self, session_id: str) -> ManagedSession:
        self._raise_if_terminal(session_id)
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        if session.state is SessionState.EXPIRED:
            raise SessionExpiredError(f"Session expired: {session_id}")
        if session.state is SessionState.FINISHED:
            raise SessionFinishedError(f"Session finished: {session_id}")
        return session

    def _force_close_session(self, session_id: str, terminal_state: SessionState) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.state = terminal_state
        session.chat.clear()
        if terminal_state is SessionState.EXPIRED:
            self._expired_sessions.add(session_id)
        if terminal_state is SessionState.FINISHED:
            self._finished_sessions.add(session_id)
