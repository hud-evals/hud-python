"""The ways out of a bounded workspace, and the policy on them.

A workspace with its own network namespace has no route anywhere — not to the
internet, and not to whatever else the substrate is running, including the
control channel that grades it. Two kinds of route are given back
deliberately: hosts on the internet, through a proxy that sees every
connection and applies the task's declared policy, and :class:`Peer` services
the environment itself runs, each forwarded to the address the task expects.

Both listen on unix sockets, so reaching them is a question of the filesystem
rather than the network: a bridge runs in the workspace's *network* namespace
while keeping the substrate's *mount* namespace, so it can see sockets the
workspace itself cannot, and offers them as ordinary ports on the workspace's
loopback. Nothing is bound into the workspace, and nothing in it can address
the substrate except through one of these.

Request parsing is the standard library's. A hand-rolled request-line parser
gets keep-alive, chunked bodies and header framing wrong in ways that surface
as a package manager failing halfway through an index rather than as an
obvious error.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.client
import json
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
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

LOGGER = logging.getLogger("hud.environment.egress")

#: In an allowlist, the entry that permits everything.
ANY_HOST = "*"

#: Headers that belong to one hop and must not be forwarded to the next.
#: ``transfer-encoding`` among them: the response body is read back already
#: de-chunked, so passing the upstream's framing along leaves the client
#: looking for chunk headers in what is now plain bytes.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "proxy-connection",
        "keep-alive",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

#: The proxy port offered on the workspace's loopback. 3128 is unremarkable —
#: an egress proxy is ordinary infrastructure, unlike a control channel.
BRIDGE_PORT = 3128

#: Where a visitor's way out is offered instead. A visitor joins the
#: workspace's network without being one of its sessions, and is held to its
#: own policy rather than the sessions' — so this is a second proxy, on a
#: second port, and it exists only while the visitor is there. Standing open
#: it would be a route the agent could take in place of the one it was given.
VISITOR_PORT = 3129

#: Run inside the workspace's network namespace, one listener per route out.
#: Its argument is ``[[host, port, socket], ...]``; every listener is bound
#: before it says it is ready, since a session that starts in between finds
#: the port refused.
_BRIDGE = """
import asyncio, json, sys

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

def bridged(path):
    async def handle(reader, writer):
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(path)
        except OSError:
            writer.close()
            return
        await asyncio.gather(splice(reader, up_writer), splice(up_reader, writer))
    return handle

async def main():
    servers = [
        await asyncio.start_server(bridged(path), host, port)
        for host, port, path in json.loads(sys.argv[1])
    ]
    print("ready", flush=True)
    await asyncio.gather(*(server.serve_forever() for server in servers))

asyncio.run(main())
"""


@dataclass(frozen=True, slots=True)
class Peer:
    """A substrate service a bounded workspace is allowed to reach.

    A workspace with its own network cannot address the substrate at all —
    that is what makes it bounded — so a service the environment runs is as
    unreachable from it as the control channel. A peer hands one of them back,
    at the address the task expects rather than wherever it happens to listen:
    ``name`` and ``port`` are what the workspace calls it, ``target`` where it
    actually answers outside (its own port on the substrate's loopback, unless
    something else is said).
    """

    name: str
    port: int
    target: tuple[str, int] | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Where the service actually listens, on the substrate."""
        return self.target or ("127.0.0.1", self.port)


def bind_addresses(peers: Sequence[Peer]) -> dict[str, str]:
    """Which loopback address each peer answers on inside the workspace.

    ``127.0.0.1`` wherever the port is free, because a task that says
    ``localhost:6379`` means that one. Two peers cannot both hold a port
    there, so the second moves down 127.0.0.0/8 and is reached by its name —
    which is how a task naming several services addresses them anyway.
    """
    taken: set[tuple[str, int]] = set()
    addresses: dict[str, str] = {}
    for peer in peers:
        if peer.name in addresses:
            raise ValueError(f"two peers are called {peer.name!r}")
        for index in range(1, 256):
            host = f"127.0.0.{index}"
            if (host, peer.port) not in taken:
                break
        else:
            raise ValueError(f"too many peers on port {peer.port}")
        taken.add((host, peer.port))
        addresses[peer.name] = host
    return addresses


def hosts_text(peers: Sequence[Peer], base: str) -> str:
    """*base* — the substrate's ``/etc/hosts`` — plus a line per peer.

    Names resolve for what runs in the workspace's *mount* namespace, which
    is its sessions. Anything joining only the network namespace (the Harbor
    verifier does, to reach a service the agent started) still reaches a peer
    at its address, but not by its name.
    """
    addresses = bind_addresses(peers)
    lines = "".join(f"{addresses[peer.name]}\t{peer.name}\n" for peer in peers)
    return f"{base.rstrip(chr(10))}\n{lines}" if base.strip() else lines


def proxy_environment(port: int, peers: Sequence[Peer] = ()) -> dict[str, str]:
    """Proxy variables for a process on a workspace's loopback.

    In the spellings clients read, and with the peers left out of them: a peer
    is reached directly, on the loopback the bridge binds it to, because sent
    through the proxy it would be resolved on the substrate, where the name
    means nothing and the address is something else. Listed one by one rather
    than as 127.0.0.0/8, which most clients (curl among them) match literally
    instead of as a network.
    """
    url = f"http://127.0.0.1:{port}"
    addresses = bind_addresses(peers)
    bypass = ",".join(dict.fromkeys(["127.0.0.1", "localhost", *addresses, *addresses.values()]))
    return {
        "http_proxy": url,
        "https_proxy": url,
        "HTTP_PROXY": url,
        "HTTPS_PROXY": url,
        "no_proxy": bypass,
        "NO_PROXY": bypass,
    }


def permitted(host: str | None, allowed: Collection[str]) -> bool:
    """Whether *host* is in *allowed*, by exact match or as a subdomain."""
    if not host:
        return False
    if ANY_HOST in allowed:
        return True
    return any(host == entry or host.endswith(f".{entry}") for entry in allowed)


def _relay(one: socket.socket, other: socket.socket, timeout: float = 300.0) -> None:
    """Copy bytes between two connected sockets until either end is done."""
    while True:
        ready, _, _ = select.select([one, other], [], [], timeout)
        if not ready:
            return
        for source in ready:
            target = other if source is one else one
            try:
                data = source.recv(65536)
                if not data:
                    return
                target.sendall(data)
            except OSError:
                return


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
        with upstream:
            _relay(self.connection, upstream)

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


class _Forward(socketserver.BaseRequestHandler):
    """One peer's socket: everything on it goes to that service, unread."""

    target: tuple[str, int] = ("127.0.0.1", 0)

    def handle(self) -> None:
        try:
            upstream = socket.create_connection(self.target, timeout=15)
        except OSError:
            return
        with upstream:
            _relay(self.request, upstream)


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        # A unix peer has no address; the handler wants one to log.
        request, _ = super().get_request()
        return request, ("workspace", 0)


class Egress:
    """A workspace's routes out, and the policy applied to them.

    ``allowed`` is the set of internet hosts a session may reach —
    ``{ANY_HOST}`` for all of them, and an empty set for a workspace that may
    reach none. ``peers`` are substrate services it may reach whatever the
    host policy says: they are named by the task rather than dialed by the
    agent, so reaching one is not a question the allowlist answers.

    Every socket lives in ``socket_dir``, which must be somewhere the
    workspace cannot see: a socket it could connect to directly would be a
    route out that skips all of this.
    """

    def __init__(
        self,
        socket_dir: Path | str,
        allowed: Collection[str],
        peers: Sequence[Peer] = (),
    ) -> None:
        self.socket_dir = Path(socket_dir)
        self.allowed = frozenset(allowed)
        self.peers = tuple(peers)
        self._servers: list[tuple[_UnixServer, Path]] = []
        self._bridge: asyncio.subprocess.Process | None = None

    @property
    def socket_path(self) -> Path:
        """The proxy's socket — the way out to the hosts policy allows."""
        return self.socket_dir / "egress.sock"

    def _peer_socket(self, index: int) -> Path:
        # By position rather than by name: a peer's name comes from the task,
        # and a task does not get to choose paths in here.
        return self.socket_dir / f"peer-{index}.sock"

    def start(self) -> None:
        """Serve the policy, and each declared peer, on a socket. Idempotent."""
        if self._servers:
            return
        self.socket_dir.mkdir(parents=True, exist_ok=True)
        if self.allowed:
            self._serve(
                self.socket_path, type("_ScopedProxy", (_Proxy,), {"allowed": self.allowed})
            )
        for index, peer in enumerate(self.peers):
            self._serve(
                self._peer_socket(index),
                type("_PeerForward", (_Forward,), {"target": peer.address}),
            )

    def _serve(self, path: Path, handler: type[socketserver.BaseRequestHandler]) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        server = _UnixServer(str(path), handler)
        os.chmod(path, 0o600)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._servers.append((server, path))

    def _bridge_spec(self, port: int) -> list[tuple[str, int, str]]:
        """Where each route out is offered inside the workspace."""
        addresses = bind_addresses(self.peers)
        return [
            *([("127.0.0.1", port, str(self.socket_path))] if self.allowed else []),
            *(
                (addresses[peer.name], peer.port, str(self._peer_socket(index)))
                for index, peer in enumerate(self.peers)
            ),
        ]

    async def attach(self, pid: int, port: int = BRIDGE_PORT) -> None:
        """Offer every route on the loopback of *pid*'s network namespace.

        The bridge joins that namespace and nothing else, so it keeps this
        filesystem — which is how it reaches sockets the workspace cannot.

        Returns once it is accepting rather than once it is spawned: a session
        starting in between finds the port refused, which a task opening with
        a package install reads as a network that does not work.
        """
        spec = self._bridge_spec(port)
        if not spec:
            return
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
                json.dumps(spec),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert self._bridge.stdout is not None
        try:
            await asyncio.wait_for(self._bridge.stdout.readline(), 30.0)
        except TimeoutError:
            LOGGER.warning("the workspace's ways out did not come up in time")

    def environment(self, port: int = BRIDGE_PORT) -> dict[str, str]:
        """Proxy variables for what this serves.

        Empty where no host is permitted: pointing a client at a proxy that
        is not there turns "this task has no network" into a connection error
        on the first hop, which reads as a broken one instead.
        """
        return proxy_environment(port, self.peers) if self.allowed else {}

    def stop(self) -> None:
        """Take the routes away."""
        if self._bridge is not None:
            with contextlib.suppress(ProcessLookupError):
                self._bridge.kill()
            self._bridge = None
        for server, path in self._servers:
            server.shutdown()
            server.server_close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)
        self._servers = []


__all__ = [
    "ANY_HOST",
    "BRIDGE_PORT",
    "VISITOR_PORT",
    "Egress",
    "Peer",
    "bind_addresses",
    "hosts_text",
    "permitted",
    "proxy_environment",
]
