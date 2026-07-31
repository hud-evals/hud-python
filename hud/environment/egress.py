"""The one way out of a bounded workspace, and the policy on it.

A workspace with its own network namespace has no route anywhere — not to the
internet, and not to whatever else the substrate is running, including the
control channel that grades it. Egress is given back deliberately, through a
proxy that sees every connection and applies the task's declared policy.

The proxy listens on a unix socket, so reaching it is a question of the
filesystem rather than the network: a bridge runs in the workspace's *network*
namespace while keeping the substrate's *mount* namespace, so it can see the
socket the workspace itself cannot, and offers it as an ordinary proxy port on
the workspace's loopback. Nothing is bound into the workspace, and nothing in
it can address the substrate.

Request parsing is the standard library's. A hand-rolled request-line parser
gets keep-alive, chunked bodies and header framing wrong in ways that surface
as a package manager failing halfway through an index rather than as an
obvious error.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import logging
import os
import select
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

LOGGER = logging.getLogger("hud.environment.egress")

#: In an allowlist, the entry that permits everything.
ANY_HOST = "*"

#: Headers that belong to one hop and must not be forwarded to the next.
_HOP_BY_HOP = frozenset(
    {"connection", "proxy-connection", "keep-alive", "te", "trailers", "upgrade"}
)

#: The proxy port offered on the workspace's loopback. 3128 is unremarkable —
#: an egress proxy is ordinary infrastructure, unlike a control channel.
BRIDGE_PORT = 3128

_BRIDGE = """
import asyncio, sys

async def splice(reader, writer):
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass

async def bridged(reader, writer):
    up_reader, up_writer = await asyncio.open_unix_connection(sys.argv[1])
    await asyncio.gather(splice(reader, up_writer), splice(up_reader, writer))

async def main():
    server = await asyncio.start_server(bridged, "127.0.0.1", int(sys.argv[2]))
    print("ready", flush=True)
    async with server:
        await server.serve_forever()

asyncio.run(main())
"""


def permitted(host: str | None, allowed: Collection[str]) -> bool:
    """Whether *host* is in *allowed*, by exact match or as a subdomain."""
    if not host:
        return False
    if ANY_HOST in allowed:
        return True
    return any(host == entry or host.endswith(f".{entry}") for entry in allowed)


class _Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    allowed: Collection[str] = ()

    def log_message(self, *_: object) -> None:
        """The workspace's traffic is not the substrate's log."""

    def _deny(self) -> None:
        # Loud and diagnosable from inside the workspace: a host held back by
        # policy should not look like a network that is merely broken.
        self.send_response(403)
        self.send_header("X-Proxy-Error", "blocked-by-allowlist")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_CONNECT(self) -> None:
        host, _, port = self.path.rpartition(":")
        if not permitted(host, self.allowed):
            self._deny()
            return
        try:
            upstream = socket.create_connection((host, int(port or 443)), timeout=15)
        except (OSError, ValueError):
            self.send_error(502)
            return
        self.send_response(200, "Connection established")
        self.end_headers()
        client = self.connection
        with upstream:
            while True:
                ready, _, _ = select.select([client, upstream], [], [], 300)
                if not ready:
                    return
                for source in ready:
                    target = upstream if source is client else client
                    try:
                        data = source.recv(65536)
                        if not data:
                            return
                        target.sendall(data)
                    except OSError:
                        return

    def _forward(self) -> None:
        parts = urllib.parse.urlsplit(self.path)
        if not permitted(parts.hostname, self.allowed):
            self._deny()
            return
        body = None
        if length := self.headers.get("Content-Length"):
            body = self.rfile.read(int(length))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
        # Rebuilt from the parsed components rather than forwarded raw: the
        # policy was applied to *this* hostname, and the request that goes out
        # must be the one it was applied to.
        path = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
        connection = http.client.HTTPConnection(parts.hostname or "", parts.port or 80, timeout=60)
        try:
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            length = response.getheader("Content-Length")
            for key, value in response.getheaders():
                if key.lower() not in _HOP_BY_HOP and key.lower() != "content-length":
                    self.send_header(key, value)
            if length is not None:
                self.send_header("Content-Length", length)
            else:
                # Nothing upstream framed the body, so the close delimits it.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            shutil.copyfileobj(response, self.wfile)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
        finally:
            connection.close()

    do_GET = _forward
    do_HEAD = _forward
    do_POST = _forward
    do_PUT = _forward
    do_DELETE = _forward
    do_PATCH = _forward
    do_OPTIONS = _forward


class _UnixProxyServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        # A unix peer has no address; the handler wants one to log.
        request, _ = super().get_request()
        return request, ("workspace", 0)


class Egress:
    """A workspace's route out, and the policy applied to it.

    ``allowed`` is the set of hosts a session may reach — ``{ANY_HOST}`` for
    all of them. An empty set is a workspace that can reach nothing, which is
    also what not starting one at all means.
    """

    def __init__(self, socket_path: Path | str, allowed: Collection[str]) -> None:
        self.socket_path = str(socket_path)
        self.allowed = frozenset(allowed)
        self._server: _UnixProxyServer | None = None
        self._thread: threading.Thread | None = None
        self._bridge: asyncio.subprocess.Process | None = None

    def start(self) -> None:
        """Serve the policy on the unix socket. Idempotent."""
        if self._server is not None:
            return
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        handler = type("_ScopedProxy", (_Proxy,), {"allowed": self.allowed})
        self._server = _UnixProxyServer(self.socket_path, handler)
        os.chmod(self.socket_path, 0o600)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    async def attach(self, pid: int, port: int = BRIDGE_PORT) -> None:
        """Offer the proxy on the loopback of *pid*'s network namespace.

        The bridge joins that namespace and nothing else, so it keeps this
        filesystem — which is how it reaches a socket the workspace cannot.

        Returns once it is accepting rather than once it is spawned: a session
        starting in between finds the port refused, which a task opening with
        a package install reads as a network that does not work.
        """
        nsenter = shutil.which("nsenter") or "/usr/bin/nsenter"
        self._bridge = await asyncio.create_subprocess_exec(
            *[
                nsenter,
                "--target",
                str(pid),
                "--net",
                "--user",
                "--preserve-credentials",
                "--",
                sys.executable,
                "-c",
                _BRIDGE,
                self.socket_path,
                str(port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert self._bridge.stdout is not None
        try:
            await asyncio.wait_for(self._bridge.stdout.readline(), 30.0)
        except TimeoutError:
            LOGGER.warning("the workspace's way out did not come up in time")

    def environment(self, port: int = BRIDGE_PORT) -> dict[str, str]:
        """Proxy variables for a session, in the spellings clients read."""
        url = f"http://127.0.0.1:{port}"
        return {
            "http_proxy": url,
            "https_proxy": url,
            "HTTP_PROXY": url,
            "HTTPS_PROXY": url,
            "no_proxy": "127.0.0.1,localhost",
            "NO_PROXY": "127.0.0.1,localhost",
        }

    def stop(self) -> None:
        """Take the route away."""
        if self._bridge is not None:
            with contextlib.suppress(ProcessLookupError):
                self._bridge.kill()
            self._bridge = None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        self._thread = None
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)


__all__ = ["ANY_HOST", "BRIDGE_PORT", "Egress", "permitted"]
