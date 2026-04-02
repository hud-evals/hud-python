"""Simple A2A client — send messages to a HUD chat server and print replies.

Unlike example 04 (which puts an LLM in front of the A2A server), this
client forwards user input directly and streams the agent's response.

Usage:
    # Terminal 1: start the A2A server
    HUD_ENV=my-assistant HUD_SCENARIO=assist HUD_MODEL=claude-haiku-4-5 \
        uv run python examples/03_a2a_chat_server.py

    # Terminal 2: run this client
    uv run python examples/05_a2a_simple_client.py

    # Or point at a different server
    A2A_URL=http://my-host:9999 uv run python examples/05_a2a_simple_client.py
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

A2A_BASE_URL = os.getenv("A2A_URL", "http://localhost:9999")
HTTP_TIMEOUT = float(os.getenv("A2A_TIMEOUT", "180"))
TERMINAL_STATES = {TaskState.completed, TaskState.failed, TaskState.canceled, TaskState.rejected}


def _extract_text(parts: Any) -> str | None:
    for part in parts or []:
        root = getattr(part, "root", part)
        text = getattr(root, "text", None)
        if text:
            return str(text)
    return None


async def create_client(http: httpx.AsyncClient) -> Any:
    resolver = A2ACardResolver(httpx_client=http, base_url=A2A_BASE_URL)
    card = await resolver.get_agent_card()
    config = ClientConfig(streaming=True, httpx_client=http)
    return ClientFactory(config=config).create(card)


async def send_and_print(
    client: Any,
    user_text: str,
    context_id: str | None,
    task_id: str | None,
) -> tuple[str | None, str | None]:
    message = Message(
        message_id=uuid.uuid4().hex,
        role=Role.user,
        parts=[Part(root=TextPart(text=user_text))],
        context_id=context_id,
        task_id=task_id,
    )

    last_context = context_id
    last_task = task_id

    async for item in client.send_message(message):
        if isinstance(item, Message):
            last_context = getattr(item, "context_id", last_context)
            last_task = getattr(item, "task_id", last_task)
            text = _extract_text(item.parts)
            if text:
                print(f"\nAgent: {text}\n")
            continue

        task, event = item
        last_context = getattr(task, "context_id", last_context)
        last_task = task.id or last_task

        if isinstance(event, TaskStatusUpdateEvent):
            if event.status.message:
                text = _extract_text(event.status.message.parts)
                if text:
                    print(f"\nAgent: {text}\n")
            if event.status.state in TERMINAL_STATES:
                last_task = None

        elif isinstance(event, TaskArtifactUpdateEvent):
            text = _extract_text(event.artifact.parts)
            if text:
                print(f"\nAgent: {text}\n")

    return last_context, last_task


async def main() -> None:
    timeout = httpx.Timeout(connect=30.0, read=HTTP_TIMEOUT, write=30.0, pool=30.0)

    async with httpx.AsyncClient(timeout=timeout) as http:
        client = await create_client(http)
        print(f"Connected to A2A server at {A2A_BASE_URL}")
        print("Type your messages below. Ctrl+C to quit.\n")

        context_id: str | None = None
        task_id: str | None = None

        while True:
            try:
                user_text = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not user_text:
                continue

            context_id, task_id = await send_and_print(client, user_text, context_id, task_id)


if __name__ == "__main__":
    asyncio.run(main())
