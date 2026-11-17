"""``Chat``: a multi-turn conversation folded over a chat-style task, one rollout per turn."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from hud import Job, Trace
from hud.eval.run import rollout
from hud.types import ContentBlock, TextContent

if TYPE_CHECKING:
    from hud import Task
    from hud.agents import Agent
    from hud.runtime import Provider

MessageContent = str | Sequence[ContentBlock]


class Chat:
    """Fold a chat-style task over a growing conversation.

    Each ``send`` reruns the task with the full history as its ``messages`` argument.
    ``messages`` is the history; persist and restore it directly.
    """

    def __init__(self, task: Task, agent: Agent, /, *, runtime: Provider) -> None:
        self._task = task
        self._agent = agent
        self._runtime = runtime
        self.messages: list[dict[str, Any]] = []
        self.job: Job | None = None

    async def send(self, message: MessageContent) -> Trace:
        """Send a user turn; the reply is the returned trace's ``content``."""
        blocks = [TextContent(type="text", text=message)] if isinstance(message, str) else message
        content = [block.model_dump() for block in blocks]
        self.messages.append(
            {"role": "user", "content": content[0] if len(content) == 1 else content}
        )

        task = self._task.model_copy(
            update={"args": {**self._task.args, "messages": list(self.messages)}},
        )
        if self.job is None:
            self.job = await Job.start(self._task.id)
        result = await rollout(task, self._agent, runtime=self._runtime, job_id=self.job.id)
        self.job.traces.append(result)
        if result.is_error:
            raise RuntimeError(result.error or "chat turn failed")

        self.messages.append(
            {"role": "assistant", "content": {"type": "text", "text": result.content or ""}}
        )
        return result
