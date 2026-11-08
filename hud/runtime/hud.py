"""HUD runtime tunnel provider."""

from __future__ import annotations

import contextlib
import logging
import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx2

from hud.settings import settings
from hud.telemetry.context import get_current_trace_id

from .core import Runtime

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud.eval.task import Task

logger = logging.getLogger("hud.runtime")

_RUNTIME_READY_TIMEOUT = 300.0


class HUDRuntime:
    """HUD tunnel placement: local agent loop against a HUD-hosted environment.

    Each acquisition leases a runtime session by environment name and yields the
    gateway's WebSocket tunnel to its control channel.
    """

    def __init__(
        self,
        *,
        runtime_url: str | None = None,
    ) -> None:
        self.runtime_url = runtime_url
        self._warned_unsupported_config = False

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return self._runtime_session(task)

    @asynccontextmanager
    async def _runtime_session(self, task: Task) -> AsyncGenerator[Runtime]:
        if task.runtime_config is not None:
            # The lease resolves the env by name, so an image or Compose recipe is only
            # provenance and cpu/memory only warn; a GPU, OS, TPU or limits change what is
            # graded, so those fail.
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
        api_key = settings.api_key
        if not api_key:
            raise RuntimeError("HUD runtime tunnel requires HUD_API_KEY")
        runtime_url = (self.runtime_url or settings.hud_runtime_url).rstrip("/")
        session_id = await self._create_runtime_session(runtime_url, api_key, task)
        try:
            yield Runtime(
                _runtime_tunnel_ws_url(runtime_url, session_id),
                params={
                    "headers": {"Authorization": f"Bearer {api_key}"},
                    "session_id": session_id,
                    "gateway_url": runtime_url,
                    "ready_timeout": _RUNTIME_READY_TIMEOUT,
                },
            )
        finally:
            await self._delete_runtime_session(runtime_url, api_key, session_id)

    async def _create_runtime_session(self, runtime_url: str, api_key: str, task: Task) -> str:
        payload: dict[str, Any] = {"environment": task.env}
        trace_id = get_current_trace_id()
        # Only a trace the platform has registered can own the session.
        if trace_id is not None and settings.reporting_enabled:
            with contextlib.suppress(ValueError):
                payload["trace_id"] = str(uuid.UUID(trace_id))
        async with httpx2.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{runtime_url}/runtime/sessions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
        if resp.is_error:
            try:
                error = resp.json()
            except ValueError:
                error = None
            detail = error.get("detail") if isinstance(error, dict) else None
            raise httpx2.HTTPStatusError(
                f"HUD runtime refused a session for environment {task.env!r} "
                f"({resp.status_code}): {detail if isinstance(detail, str) else resp.text}",
                request=resp.request,
                response=resp,
            )
        body = resp.json()
        session_id = body.get("id")
        if not isinstance(session_id, str):
            raise RuntimeError("Runtime gateway did not return a session id")
        return session_id

    async def _delete_runtime_session(
        self, runtime_url: str, api_key: str, session_id: str
    ) -> None:
        async with httpx2.AsyncClient(timeout=15.0) as client:
            with contextlib.suppress(Exception):
                await client.delete(
                    f"{runtime_url}/runtime/sessions/{session_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )


def _runtime_tunnel_ws_url(runtime_url: str, session_id: str) -> str:
    parts = urlsplit(runtime_url.rstrip("/"))
    scheme = "wss" if parts.scheme == "https" else "ws"
    path = f"{parts.path.rstrip('/')}/runtime/tunnels/{session_id}"
    return urlunsplit((scheme, parts.netloc, path, "", ""))
