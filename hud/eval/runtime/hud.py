"""HUD runtime tunnel provider."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from hud.eval.run import Run, rollout
from hud.telemetry.context import get_current_trace_id

from .core import Runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.agents.base import Agent
    from hud.eval.task import Task

logger = logging.getLogger("hud.eval.runtime")

_RUNTIME_READY_TIMEOUT = 300.0


class HUDRuntime:
    """HUD tunnel placement: local agent loop against a HUD-hosted environment.

    The SDK creates a runtime session by environment name, exposes the remote
    control channel through a local TCP listener, and lets the normal rollout
    atom drive it from this process.
    """

    def __init__(self, *, run_timeout: float = 3600.0, runtime_url: str | None = None) -> None:
        self.run_timeout = run_timeout
        self.runtime_url = runtime_url
        self._warned_unsupported_config = False

    async def run(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None = None,
        trace_id: str | None = None,
    ) -> Run:
        return await rollout(
            task,
            agent,
            runtime=self,
            trace_id=trace_id,
            job_id=job_id,
            group_id=group_id,
            rollout_timeout=self.run_timeout,
        )

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return self._runtime_session(task)

    @asynccontextmanager
    async def _runtime_session(self, task: Task) -> AsyncIterator[Runtime]:
        from hud.settings import settings as sdk_settings

        if task.runtime_config is not None:
            # The lease resolves the env by name: a stamped image is
            # provenance and rides along. Declared cpu/memory are best-effort
            # on the platform's substrate (warned once, not fatal — loaders
            # stamp them on every row), but a GPU or explicit limits change
            # what the task *is*; running without them would grade a
            # different environment than declared.
            resources = task.runtime_config.resources
            if (
                resources is not None
                and (
                    resources.gpu is not None
                    or resources.os is not None
                    or resources.tpu is not None
                )
            ) or (
                task.runtime_config.limits is not None
                and task.runtime_config.limits.model_dump(exclude_none=True)
            ):
                raise ValueError(
                    "HUDRuntime cannot honor this task's declared placement requirements or "
                    "limits on an "
                    "already-deployed env; run it on a placement that provisions them"
                )
            softly_ignored = task.runtime_config.model_dump(
                exclude_none=True, exclude={"image", "compose"}
            )
            if softly_ignored and not self._warned_unsupported_config:
                self._warned_unsupported_config = True
                logger.warning(
                    "HUDRuntime cannot honor task runtime_config %s on an "
                    "already-deployed env; rollouts proceed on the platform's "
                    "defaults",
                    sorted(softly_ignored),
                )
        api_key = sdk_settings.api_key
        if not api_key:
            raise RuntimeError("HUD runtime tunnel requires HUD_API_KEY")
        runtime_url = (self.runtime_url or sdk_settings.hud_runtime_url).rstrip("/")
        session_id = await self._create_runtime_session(runtime_url, api_key, task)
        server: asyncio.Server | None = None
        try:
            server = await asyncio.start_server(
                lambda reader, writer: self._forward_runtime_connection(
                    runtime_url,
                    api_key,
                    session_id,
                    reader,
                    writer,
                ),
                "127.0.0.1",
                0,
            )
            port = server.sockets[0].getsockname()[1]
            yield Runtime(
                f"tcp://127.0.0.1:{port}",
                params={
                    "session_id": session_id,
                    "gateway_url": runtime_url,
                    "ready_timeout": min(self.run_timeout, _RUNTIME_READY_TIMEOUT),
                },
            )
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
            await self._delete_runtime_session(runtime_url, api_key, session_id)

    async def _create_runtime_session(self, runtime_url: str, api_key: str, task: Task) -> str:
        payload: dict[str, Any] = {"environment": task.env}
        trace_id = get_current_trace_id()
        if trace_id is not None:
            with contextlib.suppress(ValueError):
                payload["trace_id"] = str(uuid.UUID(trace_id))
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{runtime_url}/runtime/sessions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        session_id = body.get("id")
        if not isinstance(session_id, str):
            raise RuntimeError("Runtime gateway did not return a session id")
        return session_id

    async def _delete_runtime_session(
        self, runtime_url: str, api_key: str, session_id: str
    ) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            with contextlib.suppress(Exception):
                await client.delete(
                    f"{runtime_url}/runtime/sessions/{session_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )

    async def _forward_runtime_connection(
        self,
        runtime_url: str,
        api_key: str,
        session_id: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        import websockets

        ws_url = _runtime_tunnel_ws_url(runtime_url, session_id)
        try:
            async with websockets.connect(
                ws_url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
                max_size=None,
            ) as websocket:
                await _splice_websocket(reader, writer, websocket)
        finally:
            if not writer.is_closing():
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()


def _runtime_tunnel_ws_url(runtime_url: str, session_id: str) -> str:
    parts = urlsplit(runtime_url.rstrip("/"))
    scheme = "wss" if parts.scheme == "https" else "ws"
    path = f"{parts.path.rstrip('/')}/runtime/tunnels/{session_id}"
    return urlunsplit((scheme, parts.netloc, path, "", ""))


async def _splice_websocket(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    websocket: Any,
) -> None:
    async def tcp_to_ws() -> None:
        while data := await reader.read(65536):
            await websocket.send(data)

    async def ws_to_tcp() -> None:
        async for message in websocket:
            data = message.encode("utf-8") if isinstance(message, str) else message
            writer.write(data)
            await writer.drain()

    tasks = [
        asyncio.create_task(tcp_to_ws()),
        asyncio.create_task(ws_to_tcp()),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        done_results = await asyncio.gather(*done, return_exceptions=True)
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    for result in done_results:
        if isinstance(result, BaseException):
            raise result
