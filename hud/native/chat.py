"""Native chat environment with sample scenarios.

Provides chat-compatible scenarios that accept ``messages`` as their
sole required parameter (typed as ``list[ContentBlock]``).

Usage::

    from hud.native.chat import env, chat_simple, chat_full
    from hud.services import Chat

    # Minimal -- no eval, just pass messages through
    chat = Chat(env("chat_simple"), model="claude-sonnet-4-5")
    r = await chat.send("What is the capital of France?")

    # Full -- rich content blocks, system prompt, eval with trace info
    chat = Chat(env("chat_full"), model="claude-sonnet-4-5")
    r = await chat.send("Analyze this data and show me a chart")

    # Serve as A2A
    chat.serve(port=9999)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud.environment import Environment
from hud.tools.types import ScenarioResult

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

env = Environment("chat")


@env.scenario()
async def chat_simple(messages: list[dict[str, Any]]) -> AsyncGenerator[Any, Any]:
    """Minimal chat -- formats messages and passes as the prompt.

    No system prompt, no evaluation logic. The agent sees the
    conversation and responds.
    """
    parts = [f"{m.get('role', 'user').title()}: {m.get('content', '')}" for m in messages]
    yield "\n".join(parts)
    yield 1.0


@env.scenario()
async def chat_full(messages: list[dict[str, Any]]) -> AsyncGenerator[Any, Any]:
    """Full-featured chat with system prompt and eval.

    Demonstrates:
    - System prompt prepended
    - Answer used in ScenarioResult with trace metadata
    """
    system = (
        "You are a helpful, accurate assistant. Use any available tools "
        "to provide thorough answers. When presenting data, structure it "
        "clearly. If you're unsure, say so."
    )

    parts = [f"System: {system}"]
    for m in messages:
        role = m.get("role", "user").title()
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("Assistant:")

    answer = yield "\n".join(parts)

    answer_str = answer if isinstance(answer, str) else str(answer)
    yield ScenarioResult(
        reward=1.0,
        content=answer_str,
        info={
            "num_messages": len(messages),
            "answer_length": len(answer_str),
        },
    )
