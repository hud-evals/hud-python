"""Scenario-only MCP server (hides raw env tools).

Run manually:
    HUD_API_KEY=... HUD_ENV_NAME=... uv run python examples/06_mcp_server.py

Required env vars:
    HUD_API_KEY                              API key for HUD
    HUD_ENV_NAME                             Environment name to connect to

Optional env vars:
    MCP_TRANSPORT=stdio|streamable-http|sse  (default: streamable-http)
    MCP_HOST=0.0.0.0                         (for HTTP/SSE)
    MCP_PORT=8765                            (for HTTP/SSE)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import hud
from openai import AsyncOpenAI
from hud.scenario_chat import run_scenario_chat_interactive
from hud.server import MCPServer

ENV_NAME = os.environ["HUD_ENV_NAME"]
TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")
HOST = os.environ.get("MCP_HOST", "0.0.0.0")
PORT = int(os.environ.get("MCP_PORT", "8765"))
SESSION_TTL_SECONDS = int(os.environ.get("HUD_SCENARIO_SESSION_TTL", "1800"))


@dataclass
class _SessionEntry:
    chat: Any
    cm: Any
    last_active: float

    def touch(self) -> None:
        self.last_active = time.monotonic()

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.last_active) > SESSION_TTL_SECONDS


mcp = MCPServer("hud-scenario-chat")
env = hud.Environment(ENV_NAME)
env.connect_hub(ENV_NAME)
sessions: dict[str, _SessionEntry] = {}
cleanup_task: asyncio.Task[Any] | None = None


async def _cleanup_expired_sessions() -> None:
    while True:
        await asyncio.sleep(60)
        expired_ids = [sid for sid, entry in sessions.items() if entry.expired]
        for sid in expired_ids:
            entry = sessions.pop(sid, None)
            if entry is None:
                continue
            try:
                await entry.chat.finish()
            except Exception:
                pass
            try:
                await entry.cm.__aexit__(None, None, None)
            except Exception:
                pass


async def _start_session(
    *,
    model: str,
    scenario: str,
    scenario_args: dict[str, Any],
    max_steps: int,
) -> tuple[str, _SessionEntry]:
    client = AsyncOpenAI(
        base_url=os.environ.get("HUD_INFERENCE_URL", "https://inference.hud.ai"),
        api_key=os.environ["HUD_API_KEY"],
    )
    cm = run_scenario_chat_interactive(
        client=client,
        model=model,
        env=env,
        scenario=scenario,
        args=scenario_args,
        max_steps=max_steps,
    )
    try:
        chat = await cm.__aenter__()
    except Exception:
        with contextlib.suppress(Exception):
            await cm.__aexit__(None, None, None)
        raise
    session_id = uuid.uuid4().hex[:16]
    entry = _SessionEntry(chat=chat, cm=cm, last_active=time.monotonic())
    sessions[session_id] = entry
    return session_id, entry


def _session_meta(session_id: str, trace_id: str) -> dict[str, str]:
    return {
        "session_id": session_id,
        "thread_id": session_id,
        "conversation_id": session_id,
        "trace_id": trace_id,
        "trace_url": f"https://hud.ai/trace/{trace_id}",
    }


@mcp.initialize
async def _initialize() -> None:
    global cleanup_task
    await env.__aenter__()
    cleanup_task = asyncio.create_task(_cleanup_expired_sessions())


@mcp.shutdown
async def _shutdown() -> None:
    global cleanup_task
    if cleanup_task is not None:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task
        cleanup_task = None

    for entry in list(sessions.values()):
        try:
            await entry.chat.finish()
        except Exception:
            pass
        try:
            await entry.cm.__aexit__(None, None, None)
        except Exception:
            pass
    sessions.clear()

    await env.__aexit__(None, None, None)


@mcp.tool()
async def scenario_list() -> dict[str, Any]:
    """List available scenarios and argument metadata."""
    scenarios = await env.list_scenarios()
    return {
        "scenarios": [
            {
                "name": s.name,
                "short_name": s.short_name,
                "description": s.description,
                "required_args": s.required_args,
                "arguments": [
                    {
                        "name": a.name,
                        "type": a.type,
                        "required": a.required,
                        "description": a.description,
                        "default": a.default,
                    }
                    for a in s.arguments
                ],
            }
            for s in scenarios
        ]
    }


@mcp.tool()
async def scenario_start(
    scenario: str,
    scenario_args: dict[str, Any],
    message: str = "Begin.",
    model: str = "gpt-4o",
    max_steps: int = 20,
) -> dict[str, Any]:
    """Start a scenario session (required first-turn bootstrap)."""
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    session_id, entry = await _start_session(
        model=model,
        scenario=scenario,
        scenario_args=scenario_args,
        max_steps=max_steps,
    )
    first_turn = await entry.chat.send(message)
    return {
        "answer": first_turn.answer,
        "tool_calls": first_turn.tool_calls,
        "hud": _session_meta(session_id, entry.chat.trace_id),
    }


@mcp.tool()
async def scenario_send(session_id: str, message: str) -> dict[str, Any]:
    """Send a follow-up turn to an existing session."""
    entry = sessions.get(session_id)
    if entry is None:
        raise ValueError(f"Session not found: {session_id}")
    entry.touch()
    turn = await entry.chat.send(message)
    return {
        "answer": turn.answer,
        "tool_calls": turn.tool_calls,
        "hud": _session_meta(session_id, entry.chat.trace_id),
    }


@mcp.tool()
async def scenario_finish(session_id: str, answer: str | None = None) -> dict[str, Any]:
    """Finish a session and return reward + trace metadata."""
    entry = sessions.get(session_id)
    if entry is None:
        raise ValueError(f"Session not found: {session_id}")
    try:
        result = await entry.chat.finish(answer=answer)
    finally:
        sessions.pop(session_id, None)
    return {
        **_session_meta(session_id, result.trace_id),
        "answer": result.answer,
        "reward": result.reward,
    }


if __name__ == "__main__":
    if TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    elif TRANSPORT in ("streamable-http", "sse", "http"):
        mcp.run(transport=TRANSPORT, host=HOST, port=PORT)
    else:
        raise ValueError(
            f"Unsupported MCP_TRANSPORT='{TRANSPORT}'. "
            "Use one of: stdio, streamable-http, sse, http."
        )