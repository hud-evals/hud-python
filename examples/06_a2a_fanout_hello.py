"""Fire concurrent hello requests to the A2A server.

Usage:
    # Ensure server is running first
    HUD_ENV=my-hud-environment uv run python examples/03_a2a_environment_orchestrator.py

    # In another terminal, run fanout test (default: 10)
    uv run python examples/06_a2a_fanout_hello.py
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass

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

A2A_URL = os.getenv("A2A_URL", "http://localhost:9999")
A2A_CARD_PATH = os.getenv("A2A_CARD_PATH", "/.well-known/agent-card.json")
A2A_CONCURRENCY = int(os.getenv("A2A_CONCURRENCY", "10"))
A2A_MESSAGE = os.getenv("A2A_MESSAGE", "hello")
A2A_TIMEOUT_SECONDS = float(os.getenv("A2A_HTTP_TIMEOUT_SECONDS", "120"))


@dataclass
class WorkerResult:
    worker_id: int
    ok: bool
    state: str
    latency_s: float
    text: str
    error: str


def _text_from_parts(parts: list[object] | None) -> str:
    if not parts:
        return ""
    chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if not text and hasattr(part, "root"):
            text = getattr(getattr(part, "root"), "text", None)
        if text:
            chunks.append(str(text))
    return "\n".join(chunks).strip()


async def _send_hello(worker_id: int) -> WorkerResult:
    started = time.perf_counter()
    timeout = httpx.Timeout(
        connect=min(20.0, A2A_TIMEOUT_SECONDS),
        read=A2A_TIMEOUT_SECONDS,
        write=20.0,
        pool=20.0,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=A2A_URL,
                agent_card_path=A2A_CARD_PATH,
            )
            card = await resolver.get_agent_card()
            client = ClientFactory(
                config=ClientConfig(streaming=True, httpx_client=httpx_client)
            ).create(card)

            message = Message(
                message_id=uuid.uuid4().hex,
                role=Role.user,
                parts=[Part(root=TextPart(text=f"{A2A_MESSAGE} (worker {worker_id})"))],
            )

            last_state = TaskState.unknown
            final_text = ""

            async for item in client.send_message(message):
                if isinstance(item, Message):
                    text = _text_from_parts(item.parts)
                    if text:
                        final_text = text
                    continue

                task, event = item
                last_state = task.status.state

                if isinstance(event, TaskArtifactUpdateEvent):
                    text = _text_from_parts(event.artifact.parts)
                    if text:
                        final_text = text

                if isinstance(event, TaskStatusUpdateEvent) and event.status.message:
                    text = _text_from_parts(event.status.message.parts)
                    if text:
                        final_text = text

            elapsed = time.perf_counter() - started
            ok = last_state in {TaskState.completed, TaskState.input_required}
            return WorkerResult(
                worker_id=worker_id,
                ok=ok,
                state=last_state.value,
                latency_s=elapsed,
                text=final_text,
                error="",
            )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return WorkerResult(
            worker_id=worker_id,
            ok=False,
            state="failed",
            latency_s=elapsed,
            text="",
            error=f"{type(exc).__name__}: {exc}",
        )


async def main() -> None:
    print(
        f"Sending {A2A_CONCURRENCY} concurrent messages to {A2A_URL} "
        f"with payload '{A2A_MESSAGE}'"
    )
    tasks = [_send_hello(i + 1) for i in range(A2A_CONCURRENCY)]
    results = await asyncio.gather(*tasks)

    success = sum(1 for r in results if r.ok)
    failure = len(results) - success
    avg_latency = sum(r.latency_s for r in results) / max(len(results), 1)

    print("\nResults:")
    for r in sorted(results, key=lambda x: x.worker_id):
        if r.ok:
            preview = (r.text or "[no text]").replace("\n", " ")[:120]
            print(
                f"  worker={r.worker_id:02d} ok state={r.state:<14} "
                f"latency={r.latency_s:.2f}s text={preview}"
            )
        else:
            print(
                f"  worker={r.worker_id:02d} FAIL state={r.state:<14} "
                f"latency={r.latency_s:.2f}s error={r.error}"
            )

    print(
        f"\nSummary: success={success}/{len(results)} "
        f"failure={failure} avg_latency={avg_latency:.2f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
