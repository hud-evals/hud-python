"""HUD-native client agent that talks to an A2A orchestrator server.

Usage:
    # Terminal 1 — start the A2A server
    HUD_ENV=my-hud-environment uv run python examples/03_a2a_environment_orchestrator.py

    # Terminal 2 — start this client agent
    uv run python examples/04_a2a_hud_native_client.py
"""

import asyncio
import os
import uuid
from typing import Any

import hud
from hud.services import Chat

A2A_URL = os.getenv("A2A_URL", "http://localhost:9999")

_A2A_CONTEXT_ID: str | None = None
_A2A_TASK_ID: str | None = None

env = hud.Environment("a2a-client")


@env.tool()
async def ask_a2a_agent(message: str) -> str:
    """Send a message to the remote A2A agent and return its response."""
    import httpx
    from a2a.client import ClientConfig, ClientFactory
    from a2a.types import Message, Part, Role, TextPart

    def _read_part_text(part: Any) -> str | None:
        return getattr(part, "text", None) or getattr(getattr(part, "root", None), "text", None)

    global _A2A_CONTEXT_ID, _A2A_TASK_ID

    response_text = ""
    input_required_text = ""
    status_updates: list[str] = []
    seen_updates: set[str] = set()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
    ) as http:
        client = await ClientFactory.connect(
            A2A_URL,
            client_config=ClientConfig(httpx_client=http),
        )

        request_kwargs: dict[str, Any] = {
            "message_id": str(uuid.uuid4()),
            "role": Role.user,
            "parts": [Part(root=TextPart(text=message))],
        }
        if _A2A_CONTEXT_ID:
            request_kwargs["context_id"] = _A2A_CONTEXT_ID
        if _A2A_TASK_ID:
            request_kwargs["task_id"] = _A2A_TASK_ID
        request = Message(**request_kwargs)

        async for item in client.send_message(request):
            maybe_msg = None
            if isinstance(item, tuple) and len(item) == 2:
                task, update = item
                _A2A_CONTEXT_ID = getattr(task, "context_id", None) or _A2A_CONTEXT_ID
                _A2A_TASK_ID = (
                    getattr(task, "task_id", None)
                    or getattr(task, "id", None)
                    or _A2A_TASK_ID
                )
                status = getattr(update, "status", None)
                state = str(getattr(status, "state", "")).lower().replace("-", "_")
                maybe_msg = getattr(status, "message", None)
                if state == "input_required" and maybe_msg is not None:
                    for part in getattr(maybe_msg, "parts", []) or []:
                        text = _read_part_text(part)
                        if text:
                            input_required_text = text
                elif maybe_msg is not None:
                    for part in getattr(maybe_msg, "parts", []) or []:
                        text = _read_part_text(part)
                        if text and text not in seen_updates:
                            seen_updates.add(text)
                            status_updates.append(text)
            else:
                maybe_msg = item
                _A2A_CONTEXT_ID = getattr(item, "context_id", None) or _A2A_CONTEXT_ID
                _A2A_TASK_ID = (
                    getattr(item, "task_id", None)
                    or getattr(item, "id", None)
                    or _A2A_TASK_ID
                )

            for part in getattr(maybe_msg, "parts", []) or []:
                text = _read_part_text(part)
                if text:
                    response_text = text

    if input_required_text:
        return f"[A2A input required] {input_required_text}"
    if not response_text:
        if status_updates:
            return "A2A status updates:\n- " + "\n- ".join(status_updates)
        return f"[A2A agent at {A2A_URL} returned no text.]"
    return response_text


@env.scenario("assist")
async def assist(messages: list | None = None):
    from mcp.types import PromptMessage, TextContent

    system = PromptMessage(
        role="user",  # type: ignore[arg-type]
        content=TextContent(
            type="text",
            text=(
                "You are a helpful client agent. When the user asks for work "
                "that requires the remote A2A agent, use the ask_a2a_agent tool. "
                "For simple questions, answer directly. Summarize remote responses."
            ),
        ),
    )

    yield [system, *(messages or [])]
    yield 1.0


async def main() -> None:
    model = os.getenv("HUD_MODEL", "gpt-4o")
    task = env("assist")
    chat = Chat(task, model=model, name="a2a-client-agent")

    print(f"Client agent ready (model={model}, A2A server={A2A_URL})")
    print("Type your messages below. Ctrl+C to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue

        trace = await chat.send(user_input)
        print(f"\nAgent: {trace.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
