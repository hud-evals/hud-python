"""Sample chat environment.

Provides chat-style tasks that accept ``messages`` as ``list[PromptMessage]``
-- each message has a role and typed content.

Serve it locally with ``hud serve chat_env.py``, or drive a task directly with
the ``Chat`` runner::

    from hud import Chat
    from hud.agents import create_agent

    chat = Chat(chat_simple(messages=[]), create_agent("claude-sonnet-4-5"))
    r = await chat.send("What is the capital of France?")
"""

from __future__ import annotations

from mcp.types import PromptMessage, TextContent

from hud.agents.types import EvaluationResult
from hud.environment import Environment

env = Environment(name="chat")


@env.template()
async def chat_simple(messages: list[PromptMessage]):
    """Minimal chat -- passes PromptMessages straight through.

    Each message keeps its role (user/assistant), so the agent's
    LLM sees proper alternating turns.
    """
    yield messages
    yield 1.0


@env.template()
async def chat_full(messages: list[PromptMessage]):
    """Full-featured chat with system prompt and eval.

    Prepends a system instruction, then passes all conversation
    messages with their original roles.
    """
    system = PromptMessage(
        role="user",  # type: ignore[arg-type]
        content=TextContent(
            type="text",
            text=(
                "You are a helpful, accurate assistant. Use any available tools "
                "to provide thorough answers. When presenting data, structure it "
                "clearly. If you're unsure, say so."
            ),
        ),
    )

    answer = yield [system, *messages]

    answer_str = answer if isinstance(answer, str) else str(answer)
    yield EvaluationResult(
        reward=1.0,
        content=answer_str,
        info={
            "num_messages": len(messages),
            "answer_length": len(answer_str),
        },
    )
