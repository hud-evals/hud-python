"""Direct A2A Python SDK client for HUD chat service servers.

Usage:
    # Terminal 1: run A2A server
    HUD_ENV=my-hud-environment HUD_SCENARIO=analysis_chat \
        uv run python examples/03_a2a_chat_server.py

    # Terminal 2: run this client
    uv run python examples/04_a2a_chat_llm_client.py

This example is intentionally more advanced than `03`: an LLM sits in front
of the A2A server and decides when to call it as a tool.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Iterable, cast

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
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
from openai import AsyncOpenAI

from hud.settings import settings

A2A_BASE_URL = os.getenv("A2A_URL", "http://localhost:9999")
A2A_CARD_PATH = os.getenv("A2A_CARD_PATH", "/.well-known/agent-card.json")
HTTP_TIMEOUT_SECONDS = float(os.getenv("A2A_HTTP_TIMEOUT_SECONDS", "180"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_LLM_TOOL_ROUNDS = int(os.getenv("MAX_LLM_TOOL_ROUNDS", "4"))
TERMINAL_TASK_STATES = {
    TaskState.completed,
    TaskState.failed,
    TaskState.canceled,
    TaskState.rejected,
}
ConversationState = dict[str, str | None]
ChatMessage = dict[str, Any]
SYSTEM_PROMPT = """You are a chat assistant with access to a backend A2A chat service.
Use the talk_to_chat_service tool when you need backend workflow knowledge or actions.
If the chat service asks follow-up questions or requests missing details, relay them clearly to the user.
Keep your final answers concise and helpful."""


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


def new_conversation_state() -> ConversationState:
    return {"context_id": None, "task_id": None}


async def create_a2a_client(httpx_client: httpx.AsyncClient):
    resolver = A2ACardResolver(
        httpx_client=httpx_client,
        base_url=A2A_BASE_URL,
        agent_card_path=A2A_CARD_PATH,
    )
    card = await resolver.get_agent_card()
    client_config = ClientConfig(streaming=True, httpx_client=httpx_client)
    return ClientFactory(config=client_config).create(card)


def create_llm_client() -> AsyncOpenAI:
    if settings.openai_api_key:
        return AsyncOpenAI(api_key=settings.openai_api_key)
    if settings.api_key:
        return AsyncOpenAI(api_key=settings.api_key, base_url=settings.hud_gateway_url)
    raise ValueError(
        "Set OPENAI_API_KEY for OpenAI, or HUD_API_KEY to use the HUD inference gateway."
    )


async def send_to_chat_service(
    client: Any,
    state: ConversationState,
    user_text: str,
) -> tuple[str, ConversationState, TaskState | None]:
    final_answer = ""
    last_state: TaskState | None = None

    message = Message(
        message_id=uuid.uuid4().hex,
        role=Role.user,
        parts=[Part(root=TextPart(text=user_text))],
        context_id=state.get("context_id"),
        task_id=state.get("task_id"),
    )

    async for item in client.send_message(message):
        if isinstance(item, Message):
            state["context_id"] = _pick_str_attr(item, "context_id", "contextId") or state.get(
                "context_id"
            )
            state["task_id"] = _pick_str_attr(item, "task_id", "taskId") or state.get("task_id")
            msg_text = _text_from_parts(item.parts)
            if msg_text:
                final_answer = msg_text
            continue

        task, event = item
        state["context_id"] = _pick_str_attr(task, "context_id", "contextId") or state.get(
            "context_id"
        )
        state["task_id"] = task.id or state.get("task_id")
        last_state = task.status.state

        if isinstance(event, TaskStatusUpdateEvent):
            if event.status.state == TaskState.failed:
                failure = _text_from_task(task)
                final_answer = failure or "Task failed."

        elif isinstance(event, TaskArtifactUpdateEvent):
            artifact_text = _text_from_parts(event.artifact.parts or [])
            if artifact_text:
                final_answer = artifact_text

        task_text = _text_from_task(task)
        if task_text:
            final_answer = task_text

    if last_state in TERMINAL_TASK_STATES:
        state["task_id"] = None

    return final_answer, state, last_state


def _tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "talk_to_chat_service",
                "description": "Send a message to the backend A2A chat service and return its reply.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the chat service.",
                        }
                    },
                    "required": ["message"],
                },
            },
        }
    ]


async def run_llm_turn(
    llm: AsyncOpenAI,
    a2a_client: Any,
    chat_service_state: ConversationState,
    llm_messages: list[ChatMessage],
) -> tuple[str, ConversationState]:
    for _ in range(MAX_LLM_TOOL_ROUNDS):
        response = await llm.chat.completions.create(
            model=LLM_MODEL,
            messages=cast(Any, llm_messages),
            tools=cast(Any, _tool_schema()),
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = cast(list[Any], list(message.tool_calls or []))

        if tool_calls:
            llm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
                }
            )
            for tool_call in tool_calls:
                args = json.loads(tool_call.function.arguments or "{}")
                tool_input = str(args.get("message", ""))
                print("  [llm] calling A2A chat service...")
                tool_output, chat_service_state, _ = await send_to_chat_service(
                    a2a_client, chat_service_state, tool_input
                )
                llm_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output or "[chat service returned no text]",
                    }
                )
            continue

        final_answer = (message.content or "").strip()
        llm_messages.append({"role": "assistant", "content": final_answer})
        return final_answer, chat_service_state

    fallback = "I hit the maximum number of backend tool calls for this turn."
    llm_messages.append({"role": "assistant", "content": fallback})
    return fallback, chat_service_state


async def main() -> None:
    timeout = httpx.Timeout(
        connect=min(30.0, HTTP_TIMEOUT_SECONDS),
        read=HTTP_TIMEOUT_SECONDS,
        write=30.0,
        pool=30.0,
    )
    llm = create_llm_client()
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        a2a_client = await create_a2a_client(httpx_client)

        print(f"LLM + A2A client ready (server={A2A_BASE_URL}, model={LLM_MODEL})")
        print("Type your messages below. Ctrl+C to quit.\n")

        chat_service_state = new_conversation_state()
        llm_messages: list[ChatMessage] = [{"role": "system", "content": SYSTEM_PROMPT}]

        while True:
            try:
                user_text = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not user_text:
                continue

            llm_messages.append({"role": "user", "content": user_text})
            final_answer, chat_service_state = await run_llm_turn(
                llm,
                a2a_client,
                chat_service_state,
                llm_messages,
            )

            if final_answer:
                print(f"\nAgent: {final_answer}\n")
            else:
                print("\nAgent: [no response]\n")


if __name__ == "__main__":
    asyncio.run(main())
