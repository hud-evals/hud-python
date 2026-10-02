"""The ways out of a bounded workspace, and the policy on them."""

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from hud.capabilities import Connection

LOGGER = logging.getLogger("hud.environment.egress")

# In an allowlist, the entry that permits everything.
ANY_HOST = "*"

# Headers for one hop only. A relayed body arrives de-chunked, so forwarding
# ``transfer-encoding`` would leave the client parsing chunk headers out of plain bytes.
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

# Dropped from relayed requests: the connection relay injects the real credential, so a
# process-side value is a placeholder that must not reach the upstream.
_CLIENT_CREDENTIALS = frozenset({"authorization", "x-api-key", "api-key", "x-goog-api-key"})


# RFC 9110 header grammar, matched positively: no control characters, so a relayed
# header can never end itself and begin another.
_FIELD_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_FIELD_VALUE = re.compile(r"[\t\x20-\x7e\x80-\xff]*")


class _Unrelayable(ValueError):
    """An upstream response that cannot be relayed as the upstream sent it."""


class _BlockedAddress(OSError):
    """A destination outside the public internet."""


def _field(name: str, value: str) -> tuple[str, str]:
    """``(name, value)`` as a header, or raise ``_Unrelayable``.

    ``http.client`` keeps a folded value's CRLF, so a verbatim header could smuggle
    others into the response. It is refused rather than silently repaired.
    """
    if not _FIELD_NAME.fullmatch(name) or not _FIELD_VALUE.fullmatch(value):
        raise _Unrelayable(f"header {name[:32]!r} is not relayable")
    return name, value


# The proxy port on the workspace's loopback (3128, the conventional proxy port).
BRIDGE_PORT = 3128

# The proxy for a visitor, which joins the workspace network without being a session and
# keeps its own policy. It exists only while the visitor does, or the agent could use it.
VISITOR_PORT = 3129

# Runs in the workspace's network namespace, one listener per route out, read from
# stdin as ``[[host, port, socket], ...]``. All are bound before "ready": a session
# starting in between would find its port refused.
_BRIDGE = """
import asyncio, json, sys

async def pump(reader, writer):
    while chunk := await reader.read(65536):
        writer.write(chunk)
        await writer.drain()
    if writer.can_write_eof():
        try:
            writer.write_eof()
        except OSError:
            pass

async def splice(a, b):
    try:
        await asyncio.gather(pump(a[0], b[1]), pump(b[0], a[1]))
    except OSError:
        pass
    finally:
        a[1].close()
        b[1].close()

def bridged(path):
    async def handle(reader, writer):
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(path)
        except OSError:
            writer.close()
            return
        await splice((reader, writer), (up_reader, up_writer))
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
    """A substrate service a bounded workspace may reach.

    ``name`` and ``port`` are what the workspace calls it; ``target`` is where it
    answers on the substrate (its own port on loopback by default).
    """

    name: str
    port: int
    target: tuple[str, int] | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Where the service actually listens, on the substrate."""
        return self.target or ("127.0.0.1", self.port)


@dataclass(frozen=True, slots=True)
class WorkspaceRoute:
    """A controller-provided host route exposed through one workspace capability."""

    capability: str
    host: str
    port: int

    def __post_init__(self) -> None:
        if not self.capability or self.capability.strip() != self.capability:
            raise ValueError("workspace route capability must not be empty or padded")
        if not self.host or self.host.strip() != self.host or any(c.isspace() for c in self.host):
            raise ValueError("workspace route host must be a hostname without whitespace")
        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            pass
        else:
            raise ValueError("workspace routes require a hostname, not an IP address")
        if not 1 <= self.port <= 65535:
            raise ValueError("workspace route port must be between 1 and 65535")

    def to_wire(self) -> dict[str, str | int]:
        return {"capability": self.capability, "host": self.host, "port": self.port}

    @classmethod
    def from_url(cls, capability: str, url: str) -> WorkspaceRoute:
        """Build a host route for one HTTP(S) endpoint."""
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in {"http", "https"} or parts.hostname is None:
            raise ValueError("workspace route URL must be HTTP(S) with a hostname")
        if parts.username is not None or parts.password is not None:
            raise ValueError("workspace route URL must not contain credentials")
        return cls(
            capability=capability,
            host=parts.hostname,
            port=parts.port or (443 if parts.scheme == "https" else 80),
        )

    @classmethod
    def from_wire(cls, value: object) -> WorkspaceRoute:
        if not isinstance(value, dict):
            raise ValueError("workspace routes must be objects")
        capability = value.get("capability")
        host = value.get("host")
        port = value.get("port")
        if not isinstance(capability, str) or not isinstance(host, str):
            raise ValueError("workspace route capability and host must be strings")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ValueError("workspace route port must be an integer")
        return cls(capability=capability, host=host, port=port)


def bind_addresses(
    peers: Sequence[Peer],
    *,
    reserved_ports: Collection[int] = (),
) -> dict[str, str]:
    """The loopback address each peer answers on inside the workspace.

    ``127.0.0.1`` while its ports are free there; a peer that would collide moves
    down 127.0.0.0/8 and is reached by name.
    """
    taken = {
        ("127.0.0.1", BRIDGE_PORT),
        ("127.0.0.1", VISITOR_PORT),
        *(("127.0.0.1", port) for port in reserved_ports),
    }
    ports_by_name: dict[str, list[int]] = {}
    for peer in peers:
        ports = ports_by_name.setdefault(peer.name, [])
        if peer.port in ports:
            raise ValueError(f"peer {peer.name!r} declares port {peer.port} twice")
        ports.append(peer.port)

    addresses: dict[str, str] = {}
    for name, ports in ports_by_name.items():
        for index in range(1, 256):
            host = f"127.0.0.{index}"
            if all((host, port) not in taken for port in ports):
                break
        else:
            raise ValueError(f"no loopback address can route peer {name!r}")
        taken.update((host, port) for port in ports)
        addresses[name] = host
    return addresses


def hosts_text(
    addresses: Mapping[str, str],
    base: str,
    *,
    local_aliases: Collection[str] = (),
) -> str:
    """``base`` (the substrate's ``/etc/hosts``) plus a line per local alias and peer.

    Only the workspace's mount namespace sees these names; a process joining just its
    network (such as the Harbor verifier) reaches peers by address only.
    """
    lines = "".join(
        [
            *(f"127.0.0.1\t{name}\n" for name in local_aliases),
            *(f"{host}\t{name}\n" for name, host in addresses.items()),
        ]
    )
    return f"{base.rstrip(chr(10))}\n{lines}" if base.strip() else lines


def proxy_environment(
    port: int,
    addresses: Mapping[str, str],
    *,
    local_aliases: Collection[str] = (),
    token: str | None = None,
) -> dict[str, str]:
    """Proxy variables for a process on a workspace's loopback.

    Peers bypass the proxy, which would resolve them on the substrate. ``no_proxy`` names
    each one because most clients, curl among them, match 127.0.0.0/8 literally.
    """
    credentials = f"hud:{urllib.parse.quote(token, safe='')}@" if token else ""
    url = f"http://{credentials}127.0.0.1:{port}"
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
    """Whether ``host`` is in ``allowed``, by exact match or as a subdomain."""
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
    """Copy bytes until both directions reach EOF, error, or the connection stalls."""
    readers = [one, other]
    while readers:
        ready, _, _ = select.select(readers, [], [], timeout)
        if not ready:
            return
        for source in ready:
            target = other if source is one else one
            try:
                data = source.recv(65536)
            except OSError:
                return
            if not data:
                readers.remove(source)
                with contextlib.suppress(OSError):
                    target.shutdown(socket.SHUT_WR)
                continue
            try:
                target.sendall(data)
            except OSError:
                return


class _Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    allowed: Collection[str] = ()
    token: str | None = None

    def log_message(self, format: str, *args: Any) -> None:
        """The workspace's traffic is not the substrate's log."""

    def _fail(self, status: int, reason: str) -> None:
        # The reason header keeps a refusal from reading as a merely broken network.
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
        # Rebuilt from the parsed URL, so the request sent is the one the policy checked.
        path = urllib.parse.urlunsplit(("", "", parts.path or "/", parts.query, ""))
        host = parts.hostname or ""
        connection = http.client.HTTPConnection(host, port, timeout=60)
        response_started = False
        try:
            connection.sock = _connect_public(host, port, timeout=60)
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
            # Validate every field before writing any: the status line commits the response.
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

    def handle(self) -> None:
        try:
            upstream = socket.create_connection(self.target, timeout=15)
        except OSError:
            return
        with upstream:
            _relay(self.request, upstream)


class _Severable(socketserver.TCPServer):
    """A stream server whose ``server_close`` also cuts the connections it accepted,
    so taking a route away ends the ones already open on it, not only new ones."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._open: set[socket.socket] = set()
        self._open_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def get_request(self) -> tuple[socket.socket, Any]:
        request, address = super().get_request()
        with self._open_lock:
            self._open.add(request)
        return request, address

    def shutdown_request(self, request: socket.socket | tuple[bytes, socket.socket]) -> None:
        assert isinstance(request, socket.socket)
        # Untracked before it closes, so server_close never shuts down a
        # descriptor that has since been reused.
        with self._open_lock:
            self._open.discard(request)
        super().shutdown_request(request)

    def server_close(self) -> None:
        super().server_close()
        with self._open_lock:
            for request in self._open:
                with contextlib.suppress(OSError):
                    request.shutdown(socket.SHUT_RDWR)


class _RelayServer(_Severable, ThreadingHTTPServer):
    daemon_threads = True


if sys.platform != "win32":  # the sockets this serves on have no Windows analogue

    class _UnixServer(_Severable, socketserver.ThreadingUnixStreamServer):
        daemon_threads = True
        request_queue_size = socket.SOMAXCONN

        def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
            # A unix peer has no address; the handler wants one to log.
            request, _ = super().get_request()
            return request, ("workspace", 0)


def _is_canonical_path(path: str) -> bool:
    """No dot segments or encoded separators an upstream could resolve out of the prefix."""
    lowered = path.lower()
    if "\\" in path or any(encoded in lowered for encoded in ("%2e", "%2f", "%5c")):
        return False
    return not any(segment in {".", ".."} for segment in path.split("/"))


class _ConnectionProxy(BaseHTTPRequestHandler):
    """Fixed-upstream reverse proxy which replaces controller-owned headers."""

    protocol_version = "HTTP/1.1"
    binding: Connection

    def log_message(self, format: str, *args: Any) -> None:
        """The bound process's traffic is not the environment's log."""

    def _fail(self, status: int, reason: str) -> None:
        self.send_response(status)
        self.send_header("X-Connection-Error", reason)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

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

    def _forward(self) -> None:
        request_target = urllib.parse.urlsplit(self.path)
        if request_target.scheme or request_target.netloc:
            self._fail(400, "absolute-request-target")
            return
        upstream = urllib.parse.urlsplit(self.binding.url)
        base_path = upstream.path.rstrip("/")
        request_path = request_target.path or "/"
        if not _is_canonical_path(request_path):
            self._fail(400, "non-canonical-path")
            return
        if base_path and request_path != base_path and not request_path.startswith(f"{base_path}/"):
            self._fail(403, "outside-connection-prefix")
            return
        try:
            body = self._request_body()
        except ValueError:
            self._fail(400, "invalid-request-body")
            return

        replacements = {
            name.casefold(): (name, value) for name, value in self.binding.headers.items()
        }
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.casefold() not in _HOP_BY_HOP
            and key.casefold() not in _CLIENT_CREDENTIALS
            and key.casefold() not in {"host", "content-length"}
            and key.casefold() not in replacements
        }
        headers.update(dict(replacements.values()))
        path = urllib.parse.urlunsplit(("", "", request_path, request_target.query, ""))
        connection_type = (
            http.client.HTTPSConnection
            if upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            upstream.hostname or "",
            upstream.port,
            timeout=300,
        )
        response_started = False
        try:
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
            relayed = [
                _field(key, value)
                for key, value in response.getheaders()
                if key.casefold() not in _HOP_BY_HOP and key.casefold() != "content-length"
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
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except _Unrelayable as error:
            LOGGER.warning("refusing to relay connection response: %s", error)
            self._fail(502, "unrelayable-upstream-header")
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


class ConnectionRelay:
    """Credential-owning HTTP relay for one controller-provided connection."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection
        handler = type(
            f"_{connection.name.title().replace('_', '')}ConnectionProxy",
            (_ConnectionProxy,),
            {"binding": connection},
        )
        self._server = _RelayServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


class Egress:
    """A workspace's routes out, and the policy applied to them.

    ``allowed`` holds the internet hosts a session may reach (``{ANY_HOST}`` for all,
    empty for none); ``peers`` are reachable whatever it says. ``socket_dir`` must be
    invisible to the workspace, or its sockets would be a route around the policy.
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
        # Fixed for the bridge's life, since it binds them once: removing a peer
        # moves no other.
        self._addresses = bind_addresses(self.peers, reserved_ports=reserved_ports)
        self.local_aliases = tuple(local_aliases)
        self.token = token
        self._servers: list[tuple[_UnixServer, Path]] = []
        self._peer_servers: dict[Peer, tuple[_UnixServer, Path]] = {}

    @property
    def addresses(self) -> Mapping[str, str]:
        """Which loopback address each peer name answers on in the workspace."""
        return self._addresses

    @property
    def socket_path(self) -> Path:
        """The proxy's socket — the way out to the hosts policy allows."""
        return self.socket_dir / "egress.sock"

    def _peer_socket(self, index: int) -> Path:
        # Indexed, not named: a peer's name comes from the task, which must not choose paths here.
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
            self._peer_servers[peer] = self._serve(
                self._peer_socket(index),
                type("_PeerForward", (_Forward,), {"target": peer.address}),
            )

    def remove_peer(self, peer: Peer) -> None:
        """Stop serving *peer*, and end the connections open to it: its bridge
        listener stays bound, and closes whatever connects to it, since there is no
        longer anything behind it."""
        server, path = self._peer_servers.pop(peer)
        self._servers.remove((server, path))
        _close(server, path)
        self.peers = tuple(candidate for candidate in self.peers if candidate != peer)
        if all(candidate.name != peer.name for candidate in self.peers):
            del self._addresses[peer.name]

    def _serve(
        self, path: Path, handler: type[socketserver.BaseRequestHandler]
    ) -> tuple[_UnixServer, Path]:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
        server = _UnixServer(str(path), handler)
        os.chmod(path, 0o600)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self._servers.append((server, path))
        return server, path

    def _bridge_spec(self, port: int) -> list[tuple[str, int, str]]:
        """Where each route out is offered inside the workspace."""
        addresses = self._addresses
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
        """Proxy variables for what this serves, or none when no host is permitted.

        A proxy that isn't there would make "no network" look like a broken one.
        """
        return (
            proxy_environment(
                port,
                self.addresses,
                local_aliases=self.local_aliases,
                token=self.token,
            )
            if self.allowed
            else {}
        )

    def stop(self) -> None:
        """Take the routes away."""
        for server, path in self._servers:
            _close(server, path)
        self._servers = []
        self._peer_servers = {}


def _close(server: _UnixServer, path: Path) -> None:
    server.shutdown()
    server.server_close()
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)
