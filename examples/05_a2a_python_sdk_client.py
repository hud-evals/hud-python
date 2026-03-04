"""Direct A2A Python SDK client for HUD orchestrator servers.

Usage:
    # Terminal 1: run A2A server
    HUD_ENV=my-hud-environment uv run python examples/03_a2a_environment_orchestrator.py

    # Terminal 2: run this client
    uv run python examples/05_a2a_python_sdk_client.py
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Iterable

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.client.errors import A2AClientJSONRPCError
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)

A2A_BASE_URL = os.getenv("A2A_URL", "http://localhost:9999")
A2A_CARD_PATH = os.getenv("A2A_CARD_PATH", "/.well-known/agent-card.json")
HTTP_TIMEOUT_SECONDS = float(os.getenv("A2A_HTTP_TIMEOUT_SECONDS", "180"))
TERMINAL_TASK_STATES = {
    TaskState.completed,
    TaskState.failed,
    TaskState.canceled,
    TaskState.rejected,
}


def _pick_str_attr(obj: object, *names: str) -> str | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, str):
            return value
    return None


def _text_from_parts(parts: Iterable[object]) -> str:
    chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if not text and hasattr(part, "root"):
            root = getattr(part, "root", None)
            text = getattr(root, "text", None) if root is not None else None
        if text:
            chunks.append(str(text))
    return "\n".join(chunks).strip()


def _text_from_task(task: Task) -> str:
    texts: list[str] = []
    if task.status and task.status.message and task.status.message.parts:
        status_text = _text_from_parts(task.status.message.parts)
        if status_text:
            texts.append(status_text)
    if task.artifacts:
        for artifact in task.artifacts:
            artifact_text = _text_from_parts(artifact.parts or [])
            if artifact_text:
                texts.append(artifact_text)
    return "\n\n".join(texts).strip()


async def main() -> None:
    timeout = httpx.Timeout(
        connect=min(30.0, HTTP_TIMEOUT_SECONDS),
        read=HTTP_TIMEOUT_SECONDS,
        write=30.0,
        pool=30.0,
    )
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=A2A_BASE_URL,
            agent_card_path=A2A_CARD_PATH,
        )
        card = await resolver.get_agent_card()

        client_config = ClientConfig(streaming=True, httpx_client=httpx_client)
        client = ClientFactory(config=client_config).create(card)

        print(f"A2A SDK client ready (server={A2A_BASE_URL})")
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

            final_answer = ""
            input_required_prompt = ""
            last_state: TaskState | None = None

            retried_without_task = False
            while True:
                message = Message(
                    message_id=uuid.uuid4().hex,
                    role=Role.user,
                    parts=[Part(root=TextPart(text=user_text))],
                    context_id=context_id,
                    task_id=None if retried_without_task else task_id,
                )
                try:
                    async for item in client.send_message(message):
                        if isinstance(item, Message):
                            context_id = (
                                _pick_str_attr(item, "context_id", "contextId")
                                or context_id
                            )
                            task_id = _pick_str_attr(item, "task_id", "taskId") or task_id
                            msg_text = _text_from_parts(item.parts)
                            if msg_text:
                                final_answer = msg_text
                            continue

                        task, event = item
                        context_id = (
                            _pick_str_attr(task, "context_id", "contextId") or context_id
                        )
                        task_id = task.id or task_id
                        last_state = task.status.state

                        if isinstance(event, TaskStatusUpdateEvent):
                            state = event.status.state
                            if state == TaskState.working and event.status.message:
                                progress = _text_from_parts(event.status.message.parts)
                                if progress:
                                    print(f"  [working] {progress}")
                            elif state == TaskState.input_required:
                                prompt = ""
                                if event.status.message:
                                    prompt = _text_from_parts(event.status.message.parts)
                                if not prompt:
                                    prompt = _text_from_task(task)
                                input_required_prompt = (
                                    prompt or "Additional input is required."
                                )
                            elif state == TaskState.failed:
                                failure = _text_from_task(task)
                                final_answer = failure or "Task failed."

                        elif isinstance(event, TaskArtifactUpdateEvent):
                            artifact_text = _text_from_parts(event.artifact.parts or [])
                            if artifact_text:
                                final_answer = artifact_text

                        task_text = _text_from_task(task)
                        if task_text:
                            final_answer = task_text
                    break
                except A2AClientJSONRPCError as exc:
                    if (
                        not retried_without_task
                        and "terminal state" in str(exc).lower()
                    ):
                        print("  [info] previous task was terminal; starting new task")
                        task_id = None
                        retried_without_task = True
                        continue
                    raise

            if input_required_prompt:
                print(f"\nA2A needs more input: {input_required_prompt}\n")
                continue

            if last_state in TERMINAL_TASK_STATES:
                task_id = None

            if final_answer:
                print(f"\nAgent: {final_answer}\n")
            elif last_state == TaskState.completed:
                print("\nAgent: [completed with no text output]\n")
            else:
                print("\nAgent: [no response]\n")


if __name__ == "__main__":
    asyncio.run(main())
