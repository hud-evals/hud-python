"""A fake Chromium DevTools endpoint: target discovery over HTTP and scripted CDP replies.

:func:`fake_browser` serves, on one loopback port like Chrome's
``--remote-debugging-port``, ``GET /json`` (the page targets plus a service
worker), ``PUT /json/new`` (a new page), and a WebSocket per page at
``/devtools/page/<id>``. Each command frame is recorded; its reply comes from
``replies[method]``: ``{"result": ...}`` or ``{"error": ...}`` is sent back,
``{"close": True}`` closes the socket instead, and ``{"hold": True}`` never
answers. Unscripted methods answer ``{"result": {}}``. Before every reply the
browser sends an unsolicited ``Page.frameNavigated`` event.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute

from .asgi import serve_asgi

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.requests import Request
    from starlette.websockets import WebSocket


@dataclass(frozen=True)
class Command:
    target: str
    method: str
    params: dict[str, Any]


@dataclass
class FakeBrowser:
    """The browser side. ``http`` and ``commands`` hold what the client sent."""

    pages: list[str] = field(default_factory=lambda: ["page-1"])
    replies: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Whether ``PUT /json/new`` creates a page; otherwise it answers ``{}``.
    can_create: bool = True
    http: list[str] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    port: int = 0

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def target(self, page: str) -> dict[str, str]:
        return {
            "id": page,
            "type": "page",
            "webSocketDebuggerUrl": f"{self.url}/devtools/page/{page}",
        }

    async def _list(self, request: Request) -> JSONResponse:
        self.http.append(f"{request.method} {request.url.path}")
        worker = {"id": "worker", "type": "service_worker", "webSocketDebuggerUrl": "ws://x"}
        return JSONResponse([worker, *(self.target(page) for page in self.pages)])

    async def _new(self, request: Request) -> JSONResponse:
        self.http.append(f"{request.method} {request.url.path}?{request.url.query}")
        if not self.can_create:
            return JSONResponse({})
        page = f"new-{len(self.pages) + 1}"
        self.pages.append(page)
        return JSONResponse(self.target(page))

    async def _session(self, websocket: WebSocket) -> None:
        page = websocket.path_params["page"]
        await websocket.accept()
        while True:
            frame = json.loads(await websocket.receive_text())
            self.commands.append(Command(page, frame["method"], frame.get("params", {})))
            reply = self.replies.get(frame["method"], {"result": {}})
            if reply.get("close"):
                await websocket.close()
                return
            if reply.get("hold"):
                continue
            event = {"method": "Page.frameNavigated", "params": {"frame": {"id": page}}}
            await websocket.send_text(json.dumps(event))
            await websocket.send_text(json.dumps({"id": frame["id"], **reply}))


@asynccontextmanager
async def fake_browser(
    *,
    pages: list[str] | None = None,
    replies: dict[str, dict[str, Any]] | None = None,
    can_create: bool = True,
) -> AsyncIterator[FakeBrowser]:
    """Serve a :class:`FakeBrowser` on a loopback port for the duration of the block."""
    browser = FakeBrowser(
        pages=["page-1"] if pages is None else list(pages),
        replies=replies or {},
        can_create=can_create,
    )
    app = Starlette(
        routes=[
            Route("/json", browser._list, methods=["GET"]),
            Route("/json/new", browser._new, methods=["PUT"]),
            WebSocketRoute("/devtools/page/{page}", browser._session),
        ]
    )
    async with serve_asgi(app) as port:
        browser.port = port
        yield browser
