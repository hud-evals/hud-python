"""A local stand-in for every HUD service the SDK talks to.

One HTTP and WebSocket server answers for the API, telemetry, inference gateway,
runtime gateway, training service and web app, each under its own path prefix.
Every request is recorded. Tests declare the replies they need with
:meth:`FakeServices.route`; anything undeclared answers 404, so a test notices
traffic it did not expect.

The server runs on its own thread and event loop, so the SDK reaches it the
same way from an in-process rollout and from a ``hud`` subprocess.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import socket
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs

import uvicorn
from starlette.applications import Starlette
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable

    from starlette.requests import Request as StarletteRequest
    from starlette.websockets import WebSocket

SERVICE_ENV = {
    "api": "HUD_API_URL",
    "telemetry": "HUD_TELEMETRY_URL",
    "gateway": "HUD_GATEWAY_URL",
    "runtime": "HUD_RUNTIME_URL",
    "rl": "HUD_RL_URL",
    "web": "HUD_WEB_URL",
}

METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


@dataclass(frozen=True)
class Request:
    """One recorded request, addressed relative to its service's prefix."""

    service: str
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes
    params: dict[str, str] = field(default_factory=dict)

    @property
    def json(self) -> Any:
        return json.loads(self.body) if self.body else None

    @property
    def bearer(self) -> str | None:
        value = self.headers.get("authorization", "")
        return value.removeprefix("Bearer ") if value.startswith("Bearer ") else None


@dataclass
class Reply:
    """What a route answers. ``stream`` sends server-sent events or other chunks as they come."""

    status: int = 200
    json: Any = None
    body: bytes | str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    content_type: str | None = None
    stream: Iterable[bytes] | None = None
    delay: float = 0.0


@dataclass
class _Route:
    service: str
    method: str
    pattern: re.Pattern[str]
    answer: Any


def _compile(path: str) -> re.Pattern[str]:
    """``/trace/{id}`` matches one segment per name; ``{rest:path}`` matches the remainder."""
    regex = re.sub(r"\{(\w+):path\}", r"(?P<\1>.*)", path)
    regex = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", regex)
    return re.compile(f"^{regex}$")


class FakeServices:
    """The fake HUD backend. Use the ``services`` fixture rather than constructing one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._routes: list[_Route] = []
        self._sockets: list[_Route] = []
        self._requests: list[Request] = []
        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self.port: int = self._socket.getsockname()[1]
        app = Starlette(
            routes=[
                Route("/{service}/{rest:path}", self._http, methods=METHODS),
                WebSocketRoute("/{service}/{rest:path}", self._websocket),
            ]
        )
        self._server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", lifespan="off", ws="websockets-sansio")
        )
        self._thread = threading.Thread(target=self._serve, name="fake-hud-services", daemon=True)

    def _serve(self) -> None:
        asyncio.run(self._server.serve(sockets=[self._socket]))

    def start(self) -> None:
        self._thread.start()
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("fake HUD services failed to start")
            threading.Event().wait(0.01)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    def reset(self) -> None:
        with self._lock:
            self._routes.clear()
            self._sockets.clear()
            self._requests.clear()

    # ─── configuration ────────────────────────────────────────────────

    def url(self, service: str) -> str:
        if service not in SERVICE_ENV:
            raise KeyError(f"unknown service {service!r}; one of {sorted(SERVICE_ENV)}")
        return f"http://127.0.0.1:{self.port}/{service}"

    def env(self) -> dict[str, str]:
        """``HUD_*_URL`` variables pointing every service here."""
        return {var: self.url(service) for service, var in SERVICE_ENV.items()}

    def route(
        self,
        service: str,
        method: str,
        path: str,
        *replies: Reply,
        handler: Callable[[Request], Reply | Awaitable[Reply]] | None = None,
        json: Any = None,
        status: int = 200,
    ) -> None:
        """Answer ``method path`` on ``service``.

        ``replies`` are served in order and the last repeats; ``handler`` computes a
        reply from each :class:`Request`; with neither, the route answers ``json``
        with ``status``. Later routes win over earlier ones, so a test can override
        a fixture's default.
        """
        if handler is not None and replies:
            raise ValueError("a route takes replies or a handler, not both")
        answer = handler or _Sequence(list(replies) or [Reply(status=status, json=json)])
        with self._lock:
            self._routes.insert(0, _Route(service, method.upper(), _compile(path), answer))

    def websocket(self, service: str, path: str, handler: Callable[..., Awaitable[None]]) -> None:
        """Serve a WebSocket at ``path``; ``handler(websocket, params)`` owns the session."""
        with self._lock:
            self._sockets.insert(0, _Route(service, "WS", _compile(path), handler))

    # ─── observation ──────────────────────────────────────────────────

    def requests(
        self, service: str | None = None, method: str | None = None, path: str | None = None
    ) -> list[Request]:
        """Recorded requests, filtered; ``path`` accepts the same ``{name}`` patterns as routes."""
        pattern = _compile(path) if path is not None else None
        with self._lock:
            recorded = list(self._requests)
        return [
            request
            for request in recorded
            if (service is None or request.service == service)
            and (method is None or request.method == method.upper())
            and (pattern is None or pattern.match(request.path))
        ]

    def bodies(self, service: str, method: str, path: str) -> list[Any]:
        return [request.json for request in self.requests(service, method, path)]

    # ─── serving ──────────────────────────────────────────────────────

    def _match(
        self, routes: list[_Route], service: str, method: str, path: str
    ) -> tuple[Any, dict[str, str]]:
        with self._lock:
            candidates = list(routes)
        for route in candidates:
            if (
                route.service == service
                and route.method == method
                and (found := route.pattern.match(path))
            ):
                return route.answer, found.groupdict()
        return None, {}

    async def _http(self, incoming: StarletteRequest) -> Response:
        service = incoming.path_params["service"]
        path = "/" + incoming.path_params["rest"]
        answer, params = self._match(self._routes, service, incoming.method, path)
        request = Request(
            service=service,
            method=incoming.method,
            path=path,
            query=parse_qs(incoming.url.query),
            headers={key.lower(): value for key, value in incoming.headers.items()},
            body=await incoming.body(),
            params=params,
        )
        with self._lock:
            self._requests.append(request)
        if answer is None:
            return Response(
                json.dumps({"detail": f"no fake route for {incoming.method} {service}{path}"}),
                status_code=404,
                media_type="application/json",
            )
        reply = answer(request)
        if inspect.isawaitable(reply):
            reply = await reply
        if reply.delay:
            await asyncio.sleep(reply.delay)
        if reply.stream is not None:
            return StreamingResponse(
                _chunks(reply.stream),
                status_code=reply.status,
                headers=reply.headers,
                media_type=reply.content_type or "text/event-stream",
            )
        if reply.body is not None:
            return Response(
                reply.body,
                status_code=reply.status,
                headers=reply.headers,
                media_type=reply.content_type or "application/octet-stream",
            )
        return Response(
            json.dumps(reply.json),
            status_code=reply.status,
            headers=reply.headers,
            media_type="application/json",
        )

    async def _websocket(self, websocket: WebSocket) -> None:
        service = websocket.path_params["service"]
        path = "/" + websocket.path_params["rest"]
        handler, params = self._match(self._sockets, service, "WS", path)
        request = Request(
            service=service,
            method="WS",
            path=path,
            query=parse_qs(websocket.url.query),
            headers={key.lower(): value for key, value in websocket.headers.items()},
            body=b"",
            params=params,
        )
        with self._lock:
            self._requests.append(request)
        if handler is None:
            await websocket.close(code=4404)
            return
        await handler(websocket, params)


class _Sequence:
    def __init__(self, replies: list[Reply]) -> None:
        if not replies:
            raise ValueError("a reply sequence needs at least one reply")
        self._replies = replies
        self._lock = threading.Lock()

    def __call__(self, request: Request) -> Reply:
        del request
        with self._lock:
            return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]


async def _chunks(stream: Iterable[bytes]) -> AsyncIterator[bytes]:
    for chunk in stream:
        yield chunk
