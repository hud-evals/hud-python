"""Chat — multi-turn conversation runner over a task.

A chat-style task takes a ``messages`` parameter and yields it as the prompt.
``Chat`` folds such a task over a growing history: each :meth:`send` appends
the user turn, drives the agent over a fresh run with the full history,
appends the reply, and returns the :class:`~hud.types.Trace`.

Example::

    from hud import Chat
    from hud.agents import create_agent
    from tasks import assistant  # an @env.task taking ``messages``

    chat = Chat(assistant(messages=[]), create_agent("claude-sonnet-4-5"))
    r1 = await chat.send("Book me a flight")
    r2 = await chat.send("SFO to JFK")

``Chat`` is protocol-agnostic: a web app, notebook, or wire protocol (A2A,
etc.) is just a frontend calling ``await chat.send(...)``. The conversation
history is the public ``messages`` list — persist and restore it directly.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from mcp.types import ContentBlock, TextContent

from hud.agents.types import AgentStep
from hud.types import Trace  # noqa: TC001 - used as return type

from .job import Job
from .rollout import rollout

if TYPE_CHECKING:
    from hud.agents.base import Agent

    from .runtime import Provider
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
    3. Drives the agent over it through the rollout engine
    4. Appends the assistant response to history
    5. Returns the Trace
    """

    def __init__(
        self,
        task: Task,
        agent: Agent,
        /,
        *,
        runtime: Provider | None = None,
    ) -> None:
        """Initialize Chat.

        Args:
            task: A :class:`hud.eval.Task` (env + task id + default args).
                Create one by calling a task, e.g. ``assistant(messages=[])``.
                Its ``messages`` arg is replaced with the running conversation
                on each :meth:`send`.
            agent: The :class:`~hud.agents.base.Agent` driving every turn
                (stateless per run, e.g. ``create_agent("claude-sonnet-4-5")``).
            runtime: Placement provider for each turn's rollout (e.g.
                ``LocalRuntime("env.py")``); defaults to HUD-hosted provisioning
                by the task's env name.
        """
        self._task = task
        self._agent = agent
        self._runtime = runtime
        self.messages: list[dict[str, Any]] = []
        #: The conversation's job — every turn's run reports under it
        #: (started on the first ``send``).
        self.job: Job | None = None

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
        # then drive the agent through the rollout engine (the chat task yields
        # these messages as the prompt; see the messages input modality).
        task = self._task.model_copy(
            update={"args": {**self._task.args, "messages": list(self.messages)}},
        )
        if self.job is None:  # one job spans the whole conversation
            self.job = await Job.start(self._task.id)
        run = await rollout(task, self._agent, runtime=self._runtime, job_id=self.job.id)
        self.job.runs.append(run)
        result = run.trace
        if result.is_error:
            # Don't record the failed turn as an assistant message.
            raise RuntimeError(result.error or "chat turn failed")

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": {"type": "text", "text": result.content or ""},
        }
        citations = result.final(lambda s: s.citations if isinstance(s, AgentStep) else None)
        if citations:
            assistant_msg["citations"] = [
                c.model_dump(mode="json", exclude_none=True) for c in citations
            ]
        self.messages.append(assistant_msg)
        return result
