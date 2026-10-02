"""Multi-turn conversation over a chat-style task."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from hud.agents.types import AgentStep
from hud.types import (
    ContentBlock,
    TextContent,
    Trace,
)

from .job import Job
from .run import rollout

if TYPE_CHECKING:
    from hud.agents.base import Agent
    from hud.runtime import Provider

    from .task import Task

LOGGER = logging.getLogger(__name__)

MessageContent = str | Sequence[ContentBlock]


def _content_to_blocks(content: MessageContent) -> list[ContentBlock]:
    if isinstance(content, str):
        return [TextContent(type="text", text=content)]
    if isinstance(content, list):
        return cast("list[ContentBlock]", content)
    return list(content)


def _blocks_to_message_content(
    blocks: Sequence[ContentBlock],
) -> dict[str, Any] | list[dict[str, Any]]:
    if len(blocks) == 1:
        return blocks[0].model_dump()
    return [block.model_dump() for block in blocks]


class Chat:
    """Fold a chat-style task over a growing conversation.

    Each ``send`` reruns the task with the full history as its ``messages``
    argument. ``messages`` is the history; persist and restore it directly.
    """

    def __init__(
        self,
        task: Task,
        agent: Agent,
        /,
        *,
        runtime: Provider,
    ) -> None:
        self._task = task
        self._agent = agent
        self._runtime = runtime
        self.messages: list[dict[str, Any]] = []
        self.job: Job | None = None

    async def send(self, message: MessageContent) -> Trace:
        """Send a user turn; the reply is the returned trace's ``content``."""
        blocks = _content_to_blocks(message)
        content_data = _blocks_to_message_content(blocks)
        self.messages.append({"role": "user", "content": content_data})

        task = self._task.model_copy(
            update={"args": {**self._task.args, "messages": list(self.messages)}},
        )
        if self.job is None:  # one job spans the whole conversation
            self.job = await Job.start(self._task.id)
        result = await rollout(task, self._agent, runtime=self._runtime, job_id=self.job.id)
        self.job.traces.append(result)
        if result.is_error:
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
