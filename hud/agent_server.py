"""Serve an HUD environment + scenarios as an OpenAI-compatible agent endpoint.

Supports multi-turn conversations (streaming and non-streaming) via session IDs.
External agents connect via standard OpenAI client -- no HUD SDK needed
on the caller side.

    from hud.agent_server import serve_agent

    env = hud.Environment("my-env")
    env.connect_hub("my-env")
    serve_agent(env, model="gpt-4o", client=AsyncOpenAI(...), port=8000)

Then any OpenAI client can call it:

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    # First turn — creates a session
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Investigate this issue"}],
        extra_body={"scenario": "my-scenario",
                    "scenario_args": {"arg": "value"}},
    )

    # Follow-up turns — reuse the session via X-HUD-Session-Id header
    session_id = r.hud["session_id"]
    r2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What are the root causes?"}],
        extra_headers={"X-HUD-Session-Id": session_id},
    )

    # Finish — submit and evaluate
    import httpx
    httpx.post(f"http://localhost:8000/v1/sessions/{session_id}/finish")
"""

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SESSION_TTL_SECONDS = 30 * 60  # 30 minutes
SESSION_ID_HEADER = "x-hud-session-id"


def _hud_meta(*, session_id: str, trace_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "thread_id": session_id,
        "conversation_id": session_id,
        "trace_id": trace_id,
        "trace_url": f"https://hud.ai/trace/{trace_id}",
    }


class _SessionEntry:
    __slots__ = ("chat", "cm", "last_active", "session_ttl")

    def __init__(self, chat: Any, cm: Any, session_ttl: int) -> None:
        self.chat = chat
        self.cm = cm
        self.session_ttl = session_ttl
        self.last_active = time.monotonic()

    def touch(self) -> None:
        self.last_active = time.monotonic()

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.last_active) > self.session_ttl


async def _cleanup_expired(sessions: dict[str, _SessionEntry]) -> None:
    """Remove sessions that have been idle longer than TTL."""
    expired = [sid for sid, entry in sessions.items() if entry.expired]
    for sid in expired:
        entry = sessions.pop(sid)
        logger.info("Session %s expired after %ds idle — cleaning up", sid, entry.session_ttl)
        try:
            await entry.chat.finish()
        except Exception:
            logger.debug("Error finishing expired session %s", sid, exc_info=True)
        try:
            await entry.cm.__aexit__(None, None, None)
        except Exception:
            logger.debug("Error closing expired session cm %s", sid, exc_info=True)


def serve_agent(
    env: Any,
    *,
    client: Any,
    model: str = "gpt-4o",
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 8000,
    api_key: str | None = None,
    workers: int = 1,
    session_ttl: int = DEFAULT_SESSION_TTL_SECONDS,
) -> None:
    """Start an OpenAI-compatible HTTP server backed by HUD scenarios.

    Args:
        env: An :class:`~hud.environment.Environment` with ``connect_hub()`` called.
        client: An ``AsyncOpenAI`` (or compatible) client for LLM calls.
        model: Default model name for completions.
        host: Bind address.
        port: Bind port.
        api_key: Optional API key to require from callers.
        workers: Number of uvicorn workers. Sessions are per-worker, so use 1
                 unless you have a sticky-session reverse proxy in front.
        session_ttl: Seconds of inactivity before a session is automatically
                     cleaned up. Defaults to 30 minutes.
    """
    import uvicorn

    if session_ttl <= 0:
        raise ValueError("session_ttl must be >= 1")

    app = _build_app(
        env=env,
        client=client,
        model=model,
        api_key=api_key,
        session_ttl=session_ttl,
    )
    uvicorn.run(app, host=host, port=port, workers=workers)


def _build_app(
    *,
    env: Any,
    client: Any,
    model: str,
    api_key: str | None,
    session_ttl: int,
) -> Any:
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field

    sessions: dict[str, _SessionEntry] = {}

    @asynccontextmanager
    async def _lifespan(_app: Any) -> AsyncIterator[None]:
        async with env:
            cleanup_task = asyncio.create_task(_cleanup_loop(sessions))
            try:
                yield
            finally:
                cleanup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cleanup_task
                for entry in sessions.values():
                    try:
                        await entry.chat.finish()
                    except Exception:
                        logger.debug("Error finishing session during shutdown", exc_info=True)
                    try:
                        await entry.cm.__aexit__(None, None, None)
                    except Exception:
                        logger.debug("Error closing session context manager", exc_info=True)
                sessions.clear()

    app = FastAPI(title="HUD Agent Server", lifespan=_lifespan)

    _default_model = model

    class ChatCompletionRequest(BaseModel):
        model: str = _default_model
        messages: list[dict[str, Any]] = Field(default_factory=list)
        stream: bool = False
        scenario: str | None = None
        scenario_args: dict[str, Any] | None = None
        thread_id: str | None = None
        conversation_id: str | None = None
        max_steps: int = Field(default=20, ge=1, le=100)

    class LifecycleToolCallRequest(BaseModel):
        name: str
        arguments: dict[str, Any] = Field(default_factory=dict)
        session_id: str | None = None
        thread_id: str | None = None
        conversation_id: str | None = None

    def _extract_session_id(
        raw_request: Request,
        *,
        session_id: str | None = None,
        thread_id: str | None = None,
        conversation_id: str | None = None,
    ) -> str | None:
        return (
            raw_request.headers.get(SESSION_ID_HEADER)
            or session_id
            or thread_id
            or conversation_id
        )

    def _hud_meta(*, session_id: str, trace_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "thread_id": session_id,
            "conversation_id": session_id,
            "trace_id": trace_id,
            "trace_url": f"https://hud.ai/trace/{trace_id}",
        }

    def _tool_schemas() -> list[dict[str, Any]]:
        return [
            {
                "name": "scenario_list",
                "description": "List available scenarios and required args.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "scenario_start",
                "description": "Create a session by starting a scenario with args and an optional first message.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scenario": {"type": "string"},
                        "scenario_args": {"type": "object"},
                        "message": {"type": "string"},
                        "model": {"type": "string"},
                        "max_steps": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["scenario", "scenario_args"],
                    "additionalProperties": True,
                },
            },
            {
                "name": "scenario_send",
                "description": "Send a follow-up user message to an existing scenario session.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "thread_id": {"type": "string"},
                        "conversation_id": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "scenario_finish",
                "description": "Finish an active session and return reward/trace metadata.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "thread_id": {"type": "string"},
                        "conversation_id": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
        ]

    async def _start_session(
        *,
        scenario: str,
        scenario_args: dict[str, Any],
        model_name: str,
        max_steps: int,
    ) -> tuple[str, _SessionEntry]:
        from hud.scenario_chat import run_scenario_chat_interactive

        session_id = uuid.uuid4().hex[:16]
        cm = run_scenario_chat_interactive(
            client=client,
            model=model_name,
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
        entry = _SessionEntry(chat=chat, cm=cm, session_ttl=session_ttl)
        sessions[session_id] = entry
        logger.info("Session %s created for scenario %s", session_id, scenario)
        return session_id, entry

    async def _finish_session(
        *,
        session_id: str,
        answer: str | None = None,
    ) -> dict[str, Any]:
        entry = sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, f"Session {session_id} not found")

        try:
            result = await entry.chat.finish(answer=answer)
        finally:
            # Always remove session from active registry; chat.finish handles cleanup.
            sessions.pop(session_id, None)

        return {
            "session_id": session_id,
            "thread_id": session_id,
            "conversation_id": session_id,
            "answer": result.answer,
            "reward": result.reward,
            "trace_id": result.trace_id,
            "trace_url": f"https://hud.ai/trace/{result.trace_id}",
        }

    async def _send_turn(
        *,
        session_id: str,
        message: str,
    ) -> dict[str, Any]:
        entry = sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, f"Session {session_id} not found")
        entry.touch()
        turn = await entry.chat.send(message)
        return {
            "session_id": session_id,
            "thread_id": session_id,
            "conversation_id": session_id,
            "answer": turn.answer,
            "trace_id": entry.chat.trace_id,
            "trace_url": f"https://hud.ai/trace/{entry.chat.trace_id}",
            "tool_calls": turn.tool_calls,
        }

    @app.middleware("http")
    async def _check_auth(request: Request, call_next: Any) -> Any:
        if api_key and request.url.path not in ("/", "/health"):
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid or missing API key"},
                )
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/scenarios")
    async def list_scenarios_route() -> dict[str, Any]:
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

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": model, "object": "model", "owned_by": "hud"}],
        }

    @app.get("/v1/lifecycle-tools")
    async def lifecycle_tools() -> dict[str, Any]:
        return {"tools": _tool_schemas()}

    @app.post("/v1/lifecycle-tools/call")
    async def lifecycle_tools_call(request: LifecycleToolCallRequest, raw: Request) -> dict[str, Any]:
        tool_name = request.name
        args = request.arguments

        if tool_name == "scenario_list":
            return await list_scenarios_route()

        if tool_name == "scenario_start":
            scenario = args.get("scenario")
            scenario_args = args.get("scenario_args")
            if not isinstance(scenario, str) or not scenario:
                raise HTTPException(400, "scenario_start requires 'scenario' (string)")
            if not isinstance(scenario_args, dict):
                raise HTTPException(400, "scenario_start requires 'scenario_args' (object)")
            message = str(args.get("message", "Begin."))
            max_steps = int(args.get("max_steps", 20))
            model_name = str(args.get("model", model))

            new_session_id, entry = await _start_session(
                scenario=scenario,
                scenario_args=scenario_args,
                model_name=model_name,
                max_steps=max_steps,
            )
            turn = await entry.chat.send(message)
            return {
                "answer": turn.answer,
                "tool_calls": turn.tool_calls,
                "hud": _hud_meta(session_id=new_session_id, trace_id=entry.chat.trace_id),
            }

        if tool_name == "scenario_send":
            session_id = _extract_session_id(
                raw,
                session_id=args.get("session_id"),
                thread_id=args.get("thread_id"),
                conversation_id=args.get("conversation_id"),
            )
            if not session_id:
                raise HTTPException(400, "scenario_send requires session_id/thread_id/conversation_id")
            message = str(args.get("message", ""))
            if not message:
                raise HTTPException(400, "scenario_send requires a non-empty 'message'")
            result = await _send_turn(session_id=session_id, message=message)
            return {
                "answer": result["answer"],
                "tool_calls": result["tool_calls"],
                "hud": _hud_meta(session_id=session_id, trace_id=result["trace_id"]),
            }

        if tool_name == "scenario_finish":
            session_id = _extract_session_id(
                raw,
                session_id=args.get("session_id"),
                thread_id=args.get("thread_id"),
                conversation_id=args.get("conversation_id"),
            )
            if not session_id:
                raise HTTPException(400, "scenario_finish requires session_id/thread_id/conversation_id")
            answer = args.get("answer")
            if answer is not None and not isinstance(answer, str):
                raise HTTPException(400, "scenario_finish 'answer' must be a string when provided")
            return await _finish_session(session_id=session_id, answer=answer)

        raise HTTPException(400, f"Unknown lifecycle tool: {tool_name}")

    @app.get("/mcp/tools")
    async def mcp_tools() -> dict[str, Any]:
        return {"tools": _tool_schemas()}

    @app.post("/mcp/tools/call")
    async def mcp_tools_call(request: LifecycleToolCallRequest, raw: Request) -> dict[str, Any]:
        # MCP-native surface maps to the same lifecycle tool handlers/runtime.
        return await lifecycle_tools_call(request, raw)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw: Request) -> Any:
        session_id = _extract_session_id(
            raw,
            thread_id=request.thread_id,
            conversation_id=request.conversation_id,
        )

        last_user_msg = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        if not last_user_msg:
            last_user_msg = "Begin."

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if session_id and session_id in sessions:
            entry = sessions[session_id]
            entry.touch()

            if request.stream:
                return StreamingResponse(
                    _stream_sse(
                        chat=entry.chat,
                        message=last_user_msg,
                        completion_id=completion_id,
                        created=created,
                        model_name=request.model,
                        session_id=session_id,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-HUD-Session-Id": session_id,
                        "X-HUD-Thread-Id": session_id,
                        "X-HUD-Trace-Id": entry.chat.trace_id,
                    },
                )

            turn = await entry.chat.send(last_user_msg)
            return _completion_response(
                completion_id=completion_id,
                created=created,
                model_name=request.model,
                content=turn.answer,
                session_id=session_id,
                trace_id=entry.chat.trace_id,
            )
        if session_id:
            raise HTTPException(404, f"Session {session_id} not found")

        if not request.scenario:
            raise HTTPException(400, "scenario is required for the first turn")
        if request.scenario_args is None:
            raise HTTPException(400, "scenario_args is required for the first turn")

        session_id, entry = await _start_session(
            scenario=request.scenario,
            scenario_args=request.scenario_args,
            model_name=request.model,
            max_steps=request.max_steps,
        )
        chat = entry.chat

        if request.stream:
            return StreamingResponse(
                _stream_sse(
                    chat=chat,
                    message=last_user_msg,
                    completion_id=completion_id,
                    created=created,
                    model_name=request.model,
                    session_id=session_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-HUD-Session-Id": session_id,
                    "X-HUD-Thread-Id": session_id,
                    "X-HUD-Trace-Id": chat.trace_id,
                },
            )

        turn = await chat.send(last_user_msg)
        return _completion_response(
            completion_id=completion_id,
            created=created,
            model_name=request.model,
            content=turn.answer,
            session_id=session_id,
            trace_id=chat.trace_id,
        )

    @app.post("/v1/sessions/{session_id}/finish")
    async def finish_session_route(session_id: str) -> dict[str, Any]:
        return await _finish_session(session_id=session_id)

    @app.get("/v1/sessions")
    async def list_sessions() -> dict[str, Any]:
        return {
            "sessions": [
                {
                    "session_id": sid,
                    "thread_id": sid,
                    "conversation_id": sid,
                    "trace_id": entry.chat.trace_id,
                    "idle_seconds": int(time.monotonic() - entry.last_active),
                }
                for sid, entry in sessions.items()
            ]
        }

    return app


async def _cleanup_loop(sessions: dict[str, _SessionEntry]) -> None:
    """Periodically evict expired sessions."""
    while True:
        await asyncio.sleep(60)
        try:
            await _cleanup_expired(sessions)
        except Exception:
            logger.debug("Session cleanup error", exc_info=True)


async def _stream_sse(
    *,
    chat: Any,
    message: str,
    completion_id: str,
    created: int,
    model_name: str,
    session_id: str,
) -> AsyncIterator[str]:
    """Yield OpenAI-format SSE chunks from a streaming scenario chat turn."""

    def _chunk(
        content: str | None = None,
        finish_reason: str | None = None,
    ) -> str:
        delta: dict[str, str] = {}
        if content is not None:
            delta["content"] = content
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(data)}\n\n"

    yield _chunk(content="")

    async for event in chat.send_stream(message):
        if event.type == "text_delta":
            yield _chunk(content=event.content)

    yield _chunk(finish_reason="stop")

    meta = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [],
        "hud": {
            "session_id": session_id,
            "trace_id": chat.trace_id,
            "trace_url": f"https://hud.ai/trace/{chat.trace_id}",
        },
    }
    yield f"data: {json.dumps(meta)}\n\n"
    yield "data: [DONE]\n\n"


def _completion_response(
    *,
    completion_id: str,
    created: int,
    model_name: str,
    content: str,
    session_id: str,
    trace_id: str,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "hud": {
            **_hud_meta(session_id=session_id, trace_id=trace_id),
        },
    }
