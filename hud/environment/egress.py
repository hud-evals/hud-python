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

import base64
import binascii
import contextlib
import hmac
import http.client
import ipaddress
import json
import logging
import os
import re
import select
import shutil
import socket
import socketserver
import sys
import threading
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import override

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
        "proxy-authenticate",
        "proxy-authorization",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


#: RFC 9110 grammar for what a header may contain. A name is a `token`; a
#: value is visible ASCII, obs-text, and blanks — no control characters, so
#: nothing in a relayed header can end it and begin another. Matched
#: positively: a proxy relays what the grammar admits, rather than guessing
#: which characters an attacker would have used.
_FIELD_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_FIELD_VALUE = re.compile(r"[\t\x20-\x7e\x80-\xff]*")


class _Unrelayable(ValueError):
    """An upstream response that cannot be relayed as the upstream sent it."""


class _BlockedAddress(OSError):
    """A destination outside the public internet."""


def _field(name: str, value: str) -> tuple[str, str]:
    """*name* and *value* as a header, or raise.

    An upstream response is remote text, and ``http.client`` preserves a
    folded value's CRLF, so a header relayed verbatim can carry headers of
    its own into the response the workspace reads. Refusing beats repairing:
    a value that has to be altered to be safe is no longer the value the
    upstream sent, and a proxy that quietly rewrites responses is worse to
    debug than one that says it could not relay this one.
    """
    if not _FIELD_NAME.fullmatch(name) or not _FIELD_VALUE.fullmatch(value):
        raise _Unrelayable(f"header {name[:32]!r} is not relayable")
    return name, value


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
#: Its stdin is ``[[host, port, socket], ...]``; every listener is bound
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
        for host, port, path in json.loads(sys.stdin.readline())
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


def bind_addresses(
    peers: Sequence[Peer],
    *,
    reserved_ports: Collection[int] = (),
) -> dict[str, str]:
    """Which loopback address each peer answers on inside the workspace.

    ``127.0.0.1`` wherever the port is free, because a task that says
    ``localhost:6379`` means that one. Two peers cannot both hold a port
    there, so the second moves down 127.0.0.0/8 and is reached by its name —
    which is how a task naming several services addresses them anyway.
    """
    taken = {
        ("127.0.0.1", BRIDGE_PORT),
        ("127.0.0.1", VISITOR_PORT),
        *(("127.0.0.1", port) for port in reserved_ports),
    }
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


def hosts_text(
    peers: Sequence[Peer],
    base: str,
    *,
    local_aliases: Collection[str] = (),
    reserved_ports: Collection[int] = (),
) -> str:
    """*base* — the substrate's ``/etc/hosts`` — plus a line per peer.

    Names resolve for what runs in the workspace's *mount* namespace, which
    is its sessions. Anything joining only the network namespace (the Harbor
    verifier does, to reach a service the agent started) still reaches a peer
    at its address, but not by its name.
    """
    addresses = bind_addresses(peers, reserved_ports=reserved_ports)
    lines = "".join(
        [
            *(f"127.0.0.1\t{name}\n" for name in local_aliases),
            *(f"{addresses[peer.name]}\t{peer.name}\n" for peer in peers),
        ]
    )
    return f"{base.rstrip(chr(10))}\n{lines}" if base.strip() else lines


def proxy_environment(
    port: int,
    peers: Sequence[Peer] = (),
    *,
    local_aliases: Collection[str] = (),
    reserved_ports: Collection[int] = (),
    token: str | None = None,
) -> dict[str, str]:
    """Proxy variables for a process on a workspace's loopback.

    In the spellings clients read, and with the peers left out of them: a peer
    is reached directly, on the loopback the bridge binds it to, because sent
    through the proxy it would be resolved on the substrate, where the name
    means nothing and the address is something else. Listed one by one rather
    than as 127.0.0.0/8, which most clients (curl among them) match literally
    instead of as a network.
    """
    credentials = f"hud:{urllib.parse.quote(token, safe='')}@" if token else ""
    url = f"http://{credentials}127.0.0.1:{port}"
    addresses = bind_addresses(peers, reserved_ports=reserved_ports)
    bypass = ",".join(
        dict.fromkeys(["127.0.0.1", "localhost", *local_aliases, *addresses, *addresses.values()])
    )
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


def _connect_public(host: str, port: int, timeout: float) -> socket.socket:
    blocked = False
    last_error: OSError | None = None
    for family, socktype, protocol, _, target in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        address = ipaddress.ip_address(target[0])
        if not address.is_global or address.is_multicast:
            blocked = True
            continue
        upstream = socket.socket(family, socktype, protocol)
        upstream.settimeout(timeout)
        try:
            upstream.connect(target)
        except OSError as error:
            upstream.close()
            last_error = error
        else:
            return upstream
    if last_error is not None:
        raise last_error
    if blocked:
        raise _BlockedAddress(f"{host} does not resolve to a public address")
    raise OSError(f"{host} has no reachable address")


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
    token: str | None = None

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """The workspace's traffic is not the substrate's log."""

    def _fail(self, status: int, reason: str) -> None:
        # Loud and diagnosable from inside the workspace: a request the proxy
        # would not carry should not look like a network that is merely broken.
        self.send_response(status)
        self.send_header("X-Proxy-Error", reason)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _deny(self) -> None:
        self._fail(403, "blocked-by-allowlist")

    def _authorized(self) -> bool:
        if self.token is None:
            return True
        value = self.headers.get("Proxy-Authorization", "")
        scheme, _, encoded = value.partition(" ")
        try:
            supplied = base64.b64decode(encoded, validate=True).decode()
        except (binascii.Error, ValueError, UnicodeDecodeError):
            supplied = ""
        if scheme.lower() == "basic" and hmac.compare_digest(supplied, f"hud:{self.token}"):
            return True
        self.send_response(407, "Proxy Authentication Required")
        self.send_header("Proxy-Authenticate", 'Basic realm="HUD verifier"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _request_body(self) -> bytes | None:
        transfer = self.headers.get("Transfer-Encoding")
        length = self.headers.get("Content-Length")
        if transfer is None:
            if length is None:
                return None
            size = int(length)
            if size < 0:
                raise ValueError
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError
            return body
        if length is not None or transfer.strip().lower() != "chunked":
            raise ValueError

        chunks: list[bytes] = []
        while True:
            line = self.rfile.readline(65537)
            if len(line) > 65536 or not line.endswith(b"\r\n"):
                raise ValueError
            size_text = line[:-2].split(b";", 1)[0].strip()
            if not size_text or any(byte not in b"0123456789abcdefABCDEF" for byte in size_text):
                raise ValueError
            size = int(size_text, 16)
            if size == 0:
                while True:
                    trailer = self.rfile.readline(65537)
                    if len(trailer) > 65536 or not trailer.endswith(b"\r\n"):
                        raise ValueError
                    if trailer == b"\r\n":
                        return b"".join(chunks)
            chunk = self.rfile.read(size)
            if len(chunk) != size or self.rfile.read(2) != b"\r\n":
                raise ValueError
            chunks.append(chunk)

    def do_CONNECT(self) -> None:
        if not self._authorized():
            return
        target = urllib.parse.urlsplit(f"//{self.path}")
        host = target.hostname
        if not permitted(host, self.allowed):
            self._deny()
            return
        try:
            upstream = _connect_public(host or "", target.port or 443, timeout=15)
        except _BlockedAddress:
            self._fail(403, "blocked-by-address")
            return
        except (OSError, ValueError):
            self.send_error(502)
            return
        self.send_response(200, "Connection established")
        self.end_headers()
        with upstream:
            _relay(self.connection, upstream)

    def _forward(self) -> None:
        if not self._authorized():
            return
        parts = urllib.parse.urlsplit(self.path)
        if not permitted(parts.hostname, self.allowed):
            self._deny()
            return
        try:
            body = self._request_body()
        except ValueError:
            self.close_connection = True
            self._fail(400, "invalid-request-body")
            return
        try:
            port = parts.port or 80
        except ValueError:
            self._fail(400, "invalid-target")
            return
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
        }
        # Rebuilt from the parsed components rather than forwarded raw: the
        # policy was applied to *this* hostname, and the request that goes out
        # must be the one it was applied to.
        path = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
        host = parts.hostname or ""
        connection = http.client.HTTPConnection(host, port, timeout=60)
        response_started = False
        try:
            connection.sock = _connect_public(host, port, timeout=60)
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
            # Validate every field before writing any of them: a response is
            # relayed whole or not at all, and half a status line has already
            # committed this connection by the time a later header fails.
            relayed = [
                _field(key, value)
                for key, value in response.getheaders()
                if key.lower() not in _HOP_BY_HOP and key.lower() != "content-length"
            ]
            length = response.getheader("Content-Length")
            framed = length is not None and length.strip().isdigit()
            _field("Reason", response.reason or "")
            response_started = True
            self.send_response(response.status, response.reason)
            for key, value in relayed:
                self.send_header(key, value)
            if framed:
                assert length is not None
                self.send_header("Content-Length", length.strip())
            else:
                # Nothing upstream framed the body, so the close delimits it.
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            shutil.copyfileobj(response, self.wfile)
        except _Unrelayable as error:
            LOGGER.warning("refusing to relay %s: %s", parts.hostname, error)
            self._fail(502, "unrelayable-upstream-header")
        except _BlockedAddress:
            self._fail(403, "blocked-by-address")
        except (OSError, http.client.HTTPException):
            if response_started:
                self.close_connection = True
            else:
                self._fail(502, "upstream-failure")
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

    @override
    def handle(self) -> None:
        try:
            upstream = socket.create_connection(self.target, timeout=15)
        except OSError:
            return
        with upstream:
            _relay(self.request, upstream)


class _UnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    @override
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
        *,
        local_aliases: Collection[str] = (),
        reserved_ports: Collection[int] = (),
        token: str | None = None,
    ) -> None:
        self.socket_dir = Path(socket_dir)
        self.allowed = frozenset(allowed)
        self.peers = tuple(peers)
        self.local_aliases = tuple(local_aliases)
        self.reserved_ports = frozenset(reserved_ports)
        self.token = token
        self._servers: list[tuple[_UnixServer, Path]] = []

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
                self.socket_path,
                type(
                    "_ScopedProxy",
                    (_Proxy,),
                    {"allowed": self.allowed, "token": self.token},
                ),
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
        addresses = bind_addresses(self.peers, reserved_ports=self.reserved_ports)
        return [
            *([("127.0.0.1", port, str(self.socket_path))] if self.allowed else []),
            *(
                (addresses[peer.name], peer.port, str(self._peer_socket(index)))
                for index, peer in enumerate(self.peers)
            ),
        ]

    def bridge_command(
        self,
        port: int = BRIDGE_PORT,
        *,
        visitor_socket: Path | None = None,
    ) -> tuple[list[str], bytes]:
        """Command and private input serving this policy in the workspace network."""
        routes = self._bridge_spec(port)
        if visitor_socket is not None:
            routes.append(("127.0.0.1", VISITOR_PORT, str(visitor_socket)))
        config = json.dumps(routes).encode() + b"\n"
        return [sys.executable, "-c", _BRIDGE], config

    def environment(self, port: int = BRIDGE_PORT) -> dict[str, str]:
        """Proxy variables for what this serves.

        Empty where no host is permitted: pointing a client at a proxy that
        is not there turns "this task has no network" into a connection error
        on the first hop, which reads as a broken one instead.
        """
        return (
            proxy_environment(
                port,
                self.peers,
                local_aliases=self.local_aliases,
                reserved_ports=self.reserved_ports,
                token=self.token,
            )
            if self.allowed
            else {}
        )

    def stop(self) -> None:
        """Take the routes away."""
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
