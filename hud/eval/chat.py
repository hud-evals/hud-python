"""Chat — multi-turn conversation runner over a task.

A chat-style task takes a ``messages`` parameter and yields it as the prompt.
``Chat`` folds such a task over a growing history: each :meth:`send` appends
the user turn, drives a fresh agent over a fresh run with the full history,
appends the reply, and returns the :class:`~hud.types.Trace`.

Example::

    from hud import Chat
    from tasks import assistant  # an @env.task taking ``messages``

    chat = Chat(assistant(messages=[]), model="claude-sonnet-4-5")
    r1 = await chat.send("Book me a flight")
    r2 = await chat.send("SFO to JFK")

``Chat`` is protocol-agnostic: a web app, notebook, or wire protocol (A2A,
etc.) is just a frontend calling ``await chat.send(...)``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from mcp.types import ContentBlock, TextContent

from hud.types import Trace  # noqa: TC001 - used as return type

if TYPE_CHECKING:
    from .task import Task

LOGGER = logging.getLogger(__name__)

MessageContent = str | Sequence[ContentBlock]


def _content_to_blocks(content: MessageContent) -> list[ContentBlock]:
    """Normalize message content to a list of ContentBlocks."""
    if isinstance(content, str):
        return [TextContent(type="text", text=content)]
    if isinstance(content, list):
        return cast("list[ContentBlock]", content)
    return list(content)


def _blocks_to_message_content(
    blocks: Sequence[ContentBlock],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Serialize blocks for PromptMessage-compatible `content`.

    Preserve multi-block inputs instead of silently dropping blocks.
    """
    if len(blocks) == 1:
        return blocks[0].model_dump()
    return [block.model_dump() for block in blocks]


class Chat:
    """Fold a chat-style task over a conversation history.

    Each ``send()`` call:
    1. Appends the user message to history
    2. Creates a Task copy with the full history as the ``messages`` arg
    3. Enters the Task, lets the agent drive the Run, then grades on exit
    4. Appends the assistant response to history
    5. Returns the Trace
    """

    def __init__(
        self,
        task: Task,
        /,
        *,
        model: str,
        agent_params: dict[str, Any] | None = None,
        max_steps: int = 10,
    ) -> None:
        """Initialize Chat.

        Args:
            task: A :class:`hud.eval.Task` (env + task id + default args).
                Positional only. Create one by calling a task, e.g.
                ``assistant(messages=[])``. Its ``messages`` arg is replaced with
                the running conversation on each :meth:`send`.
            model: Model name string (e.g. "claude-sonnet-4-5").
                Auto-resolves to the right agent via the HUD gateway.
            agent_params: Extra kwargs forwarded to agent creation
            max_steps: Max agent tool-call steps per turn
        """
        self._task = task
        self._model = model
        self._agent_params = agent_params or {}
        self._max_steps = max_steps
        self.messages: list[dict[str, Any]] = []

    def _create_agent(self) -> Any:
        """Create an agent instance from the configured model name."""
        from hud.agents import create_agent

        return create_agent(self._model, **{"max_steps": self._max_steps, **self._agent_params})

    async def send(self, message: MessageContent) -> Trace:
        """Send a user message and get the agent's response.

        Args:
            message: Plain text string or list of ContentBlocks

        Returns:
            Trace with the agent's response in ``trace.content``
        """
        blocks = _content_to_blocks(message)

        # Build PromptMessage-compatible content (single block dict or block list)
        content_data = _blocks_to_message_content(blocks)

        self.messages.append({"role": "user", "content": content_data})

        # Rebuild the task with the running conversation as the ``messages`` arg,
        # then drive the agent over a fresh run (the chat task yields these messages
        # as the prompt; see the messages input modality).
        task = replace(
            self._task,
            args={**self._task.args, "messages": list(self.messages)},
        )
        agent = self._create_agent()
        async with task as run:
            await agent(run)
        result = run.trace

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": {"type": "text", "text": result.content or ""},
        }
        if result.citations:
            assistant_msg["citations"] = result.citations
        self.messages.append(assistant_msg)
        return result

    def clear(self) -> None:
        """Reset the conversation history."""
        self.messages = []

    def export_history(self) -> list[dict[str, Any]]:
        """Export the conversation history for persistence.

        Returns a JSON-serializable list of message dicts that can be
        saved and later restored with ``load_history()``.
        """
        return [dict(m) for m in self.messages]

    def load_history(self, messages: list[dict[str, Any]]) -> None:
        """Restore conversation history from a previous export.

        Replaces the current history. Use after ``export_history()`` to
        resume a conversation across server restarts or sessions.
        """
        self.messages = [dict(m) for m in messages]
