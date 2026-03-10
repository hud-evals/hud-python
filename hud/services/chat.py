"""Chat -- unified agent runner for multi-turn, tools, and A2A.

Subclasses A2A ``AgentExecutor`` so it can be plugged directly into
``DefaultRequestHandler``.  Also works standalone for multi-turn
conversations and can produce MCP tools.

Example::

    from hud.services import Chat
    from hud.eval import Task

    chat = Chat(
        Task(env={"name": "my-env"}, scenario="analysis_chat"),
        model="claude-sonnet-4-20250514",
    )

    # Multi-turn conversation
    r1 = await chat.send("Book me a flight")
    r2 = await chat.send("SFO to JFK")

    # As MCP tool for another agent
    tool = chat.as_tool()

    # Serve as A2A endpoint
    chat.serve(port=9999)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import AgentExecutor
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message,
    Part,
    Role,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from mcp.types import ContentBlock, TextContent

from hud.services.reply_metadata import build_reply_metadata_event
from hud.types import Trace  # noqa: TC001 - used as return type

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue

    from hud.eval.task import Task
    from hud.tools.agent import AgentTool

LOGGER = logging.getLogger(__name__)

MessageContent = str | Sequence[ContentBlock]


def _content_to_blocks(content: MessageContent) -> list[ContentBlock]:
    """Normalize message content to a list of ContentBlocks."""
    if isinstance(content, str):
        return [TextContent(type="text", text=content)]
    if isinstance(content, list):
        return content  # type: ignore[return-value]
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


class Chat(AgentExecutor):
    """Unified agent runner: multi-turn chat, MCP tool, and A2A executor.

    Each ``send()`` call:
    1. Appends the user message to history
    2. Creates a Task copy with the full history as scenario args
    3. Runs ``hud.eval(task)`` -> scenario setup -> ``agent.run(ctx)`` -> evaluate
    4. Appends the assistant response to history
    5. Returns the Trace

    Subclasses ``AgentExecutor`` from the A2A SDK so it can be plugged
    directly into ``DefaultRequestHandler``.
    """

    def __init__(
        self,
        task: Task,
        /,
        *,
        model: str,
        agent_params: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        max_steps: int = 10,
        trace: bool = True,
        quiet: bool = True,
    ) -> None:
        """Initialize Chat.

        Args:
            task: Task template (env + scenario + default args).
                Positional only. Use ``env("scenario")`` or
                ``scenario_handle.task()`` to create one.
            model: Model name string (e.g. "claude-sonnet-4-20250514").
                Auto-resolves to the right agent class.
            agent_params: Extra kwargs forwarded to agent creation
            name: Human-readable name for AgentCard generation
            description: Description for AgentCard generation
            trace: Whether to record traces on the HUD platform
            quiet: When True, suppress banner/link output (default for chat)
        """
        self._task = task
        self._model = model
        self._agent_params = agent_params or {}
        self._name = name or task.scenario or "chat"
        self._description = description or f"Chat agent for {task.scenario or 'tasks'}"
        self._max_steps = max_steps
        self._trace = trace
        self._quiet = quiet
        self.messages: list[dict[str, Any]] = []

    def _create_agent(self) -> Any:
        """Create an agent instance from the configured model name."""
        from hud.agents import create_agent

        return create_agent(self._model, **self._agent_params)

    # ------------------------------------------------------------------
    # Direct usage
    # ------------------------------------------------------------------

    async def send(self, message: MessageContent) -> Trace:
        """Send a user message and get the agent's response.

        Args:
            message: Plain text string or list of ContentBlocks

        Returns:
            Trace with the agent's response in ``trace.content``
        """
        import hud

        blocks = _content_to_blocks(message)

        # Build PromptMessage-compatible content (single block dict or block list)
        content_data = _blocks_to_message_content(blocks)

        self.messages.append({"role": "user", "content": content_data})

        task_args = dict(self._task.args or {})
        task_args["messages"] = self.messages
        task = self._task.model_copy(update={"args": task_args})

        async with hud.eval(task, trace=self._trace, quiet=self._quiet) as ctx:
            agent = self._create_agent()
            result = await agent.run(ctx, max_steps=self._max_steps)

        self.messages.append(
            {
                "role": "assistant",
                "content": {"type": "text", "text": result.content or ""},
            }
        )
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

    # ------------------------------------------------------------------
    # MCP tool surface
    # ------------------------------------------------------------------

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> AgentTool:
        """Return an AgentTool backed by this Chat's config."""
        from hud.tools.agent import AgentTool

        return AgentTool(
            self._task,
            model=self._model,
            agent_params=self._agent_params,
            name=name,
            description=description,
        )

    # ------------------------------------------------------------------
    # A2A serving
    # ------------------------------------------------------------------

    def agent_card(self, url: str = "http://localhost:9999/") -> AgentCard:
        """Generate an AgentCard from this Chat's configuration."""
        skills = [
            AgentSkill(
                id=self._task.scenario or "default",
                name=self._name,
                description=self._description,
                tags=[self._task.scenario or "chat"],
            )
        ]

        return AgentCard(
            name=self._name,
            description=self._description,
            url=url,
            version="1.0",
            capabilities=AgentCapabilities(streaming=True),
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            skills=skills,
        )

    def serve(
        self,
        *,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 9999,
        url: str | None = None,
    ) -> None:
        """Start an A2A server serving this Chat.

        Blocks until interrupted. Uses Uvicorn as the ASGI server.

        Args:
            host: Bind address
            port: Bind port
            url: Public URL for the AgentCard (auto-generated if not provided)
        """
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

        LOGGER.info("Serving A2A agent at %s", public_url)
        uvicorn.run(app.build(), host=host, port=port)

    # ------------------------------------------------------------------
    # A2A AgentExecutor interface
    # ------------------------------------------------------------------

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Process an A2A message via send()."""
        context_id = context.context_id or str(uuid.uuid4())
        task_id = context.task_id or str(uuid.uuid4())

        try:
            message_text = context.get_user_input()

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    context_id=context_id,
                    task_id=task_id,
                    final=False,
                    status=TaskStatus(
                        state=TaskState.working,
                    ),
                )
            )

            result = await self.send(message_text)
            content = result.content or ""
            metadata_event = build_reply_metadata_event(
                context_id=context_id,
                task_id=task_id,
                trace=result,
            )
            if metadata_event is not None:
                await event_queue.enqueue_event(metadata_event)

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    context_id=context_id,
                    task_id=task_id,
                    final=True,
                    status=TaskStatus(
                        state=TaskState.input_required,
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=content))],
                        ),
                    ),
                )
            )
        except Exception as exc:
            LOGGER.exception("Chat A2A execute failed")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    context_id=context_id,
                    task_id=task_id,
                    final=True,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=Message(
                            message_id=str(uuid.uuid4()),
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=str(exc)))],
                        ),
                    ),
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing task and clear conversation history."""
        context_id = context.context_id or ""
        task_id = context.task_id or ""

        self.clear()

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                context_id=context_id,
                task_id=task_id,
                final=True,
                status=TaskStatus(state=TaskState.canceled),
            )
        )
