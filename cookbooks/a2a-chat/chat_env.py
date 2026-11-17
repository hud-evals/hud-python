"""Sample chat environment.

Provides chat-style tasks that accept ``messages`` as ``list[PromptMessage]``:
each message has a role and typed content. The agent sees the conversation as a
transcript, one labelled turn per message.

Serve it locally with ``hud serve chat_env.py``, or drive a task directly with
the ``Chat`` runner in ``chat.py``:

    from chat import Chat
    from hud.agents import ClaudeCLIAgent
    from hud.agents.types import ClaudeCLIConfig
    from hud.runtime import LocalRuntime

    chat = Chat(
        chat_simple(messages=[]),
        ClaudeCLIAgent(ClaudeCLIConfig(model="claude-haiku-4-5", gateway=True)),
        runtime=LocalRuntime("chat_env.py"),
    )
    r = await chat.send("What is the capital of France?")
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hud import Environment
from hud.graders import EvaluationResult
from hud.types import PromptMessage, TextContent

env = Environment(name="chat")
# CLI agents run in a workspace; this one is scratch space for the agent.
env.workspace(Path(tempfile.mkdtemp(prefix="chat-")))


def transcript(messages: list[PromptMessage]) -> str:
    """The conversation as labelled turns, since a CLI agent reads one text prompt."""
    turns = []
    for message in messages:
        text = message.content.text if isinstance(message.content, TextContent) else "[non-text]"
        turns.append(f"{message.role.capitalize()}: {text}")
    return "\n\n".join(turns)


@env.template()
async def chat_simple(messages: list[PromptMessage]):
    """Minimal chat: the conversation so far, for the agent to answer."""
    yield transcript(messages)
    yield 1.0


@env.template()
async def chat_full(messages: list[PromptMessage]):
    """Full-featured chat: an instruction, then the conversation, then a graded reply."""
    instruction = (
        "You are a helpful, accurate assistant. Use any available tools to provide thorough "
        "answers. When presenting data, structure it clearly. If you're unsure, say so. "
        "Reply to the last user turn of this conversation."
    )
    answer = yield f"{instruction}\n\n{transcript(messages)}"

    answer_str = answer or ""
    yield EvaluationResult(
        reward=1.0,
        content=answer_str,
        info={
            "num_messages": len(messages),
            "answer_length": len(answer_str),
        },
    )
