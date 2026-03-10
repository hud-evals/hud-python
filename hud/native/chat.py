"""Native chat environment with sample scenarios.

Provides chat-compatible scenarios that accept ``messages`` as
``list[PromptMessage]`` -- each message has a role and typed content.

Usage::

    from hud.native.chat import env

    chat = env.chat("chat_simple", model="claude-sonnet-4-5")
    r = await chat.send("What is the capital of France?")

    chat = env.chat("chat_full", model="claude-sonnet-4-5")
    r = await chat.send("Analyze this data")

    chat.serve(port=9999)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.types import PromptMessage, TextContent

from hud.environment import Environment
from hud.tools.types import ScenarioResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

env = Environment("chat")


@env.scenario()
async def chat_simple(messages: list[PromptMessage]) -> AsyncGenerator[Any, Any]:
    """Minimal chat -- passes PromptMessages straight through.

    Each message keeps its role (user/assistant), so the agent's
    LLM sees proper alternating turns.
    """
    yield messages
    yield 1.0


@env.scenario()
async def chat_full(messages: list[PromptMessage]) -> AsyncGenerator[Any, Any]:
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
    yield ScenarioResult(
        reward=1.0,
        content=answer_str,
        info={
            "num_messages": len(messages),
            "answer_length": len(answer_str),
        },
    )
