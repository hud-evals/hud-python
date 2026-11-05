"""A bounded workspace's ways out: the policy proxy, peer forwards and the bridge.

The proxy and each peer forward serve on a unix socket, and the bridge offers
them as ports inside the workspace's network. Scenarios speak to them over real
sockets; the internet behind the proxy is :func:`internet`.
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import ipaddress
import json
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud.capabilities import Connection
from hud.environment.egress import (
    ANY_HOST,
    ConnectionRelay,
    Egress,
    Peer,
)
from tests.conftest import eventually

from .conftest import shell

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

REQUEST = (
    b"POST /v1/echo HTTP/1.1\r\nHost: upstream\r\nContent-Length: 12\r\n"
    b"Connection: close\r\n\r\nrequest body"
)
BODY = b"complete response" * 32768
RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s" % (
    len(BODY),
    BODY,
)


def internet(
    monkeypatch: pytest.MonkeyPatch,
    servers: Mapping[str, int] | None = None,
    resolve: Mapping[str, str] | None = None,
) -> None:
    """Stand in for the internet, as a process in the test reaches it by name, until the
    test ends.

    Each name in ``servers`` resolves to a public address of its own, and a connection
    to that address reaches the local port the test serves the host on, so code that
    refuses anything but the public internet still reaches a test server. A name in
    ``resolve`` resolves to the address given. Every other name fails to resolve, as it
    would offline; IP literals and ``localhost`` resolve as usual.
    """
    public = {name: f"6.0.0.{index}" for index, name in enumerate(servers or {}, start=1)}
    ports = {public[name]: port for name, port in (servers or {}).items()}
    addresses = {**public, **(resolve or {})}
    getaddrinfo = socket.getaddrinfo
    connect = socket.socket.connect

    def lookup(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        name = host.decode() if isinstance(host, bytes) else host
        if name in addresses:
            return getaddrinfo(addresses[name], port, *args, **kwargs)
        try:
            ipaddress.ip_address(name)
        except ValueError:
            if name not in (None, "localhost"):
                raise socket.gaierror(socket.EAI_NONAME, f"{name} does not resolve") from None
        return getaddrinfo(host, port, *args, **kwargs)

    def dial(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address[0] in ports:
            address = ("127.0.0.1", ports[address[0]])
        connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setattr(socket.socket, "connect", dial)


@contextmanager
def sockets() -> Iterator[Path]:
    """A socket directory outside the pytest tmp dir: unix socket paths cap near 104 bytes."""
    directory = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@contextmanager
def proxy(allowed: set[str], **options: Any) -> Iterator[Egress]:
    with sockets() as directory:
        egress = Egress(directory, allowed, **options)
        egress.start()
        try:
            yield egress
        finally:
            egress.stop()


def exchange(path: Path, request: bytes) -> bytes:
    """One request to a unix-socket proxy, and its whole response."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(5)
        client.connect(str(path))
        client.sendall(request)
        response = b""
        while b"\r\n\r\n" not in response:
            response += client.recv(65536)
        head, _, body = response.partition(b"\r\n\r\n")
        lengths = [
            int(line.split(b":", 1)[1])
            for line in head.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        ]
        if not lengths:
            return response + drain(client)
        while len(body) < lengths[0]:
            body += client.recv(65536)
        return head + b"\r\n\r\n" + body


def basic(token: str) -> str:
    return base64.b64encode(f"hud:{token}".encode()).decode()


def drain(client: socket.socket) -> bytes:
    received = bytearray()
    while chunk := client.recv(65536):
        received.extend(chunk)
    return bytes(received)


@pytest.mark.parametrize(
    ("allowed", "token", "request_bytes", "status", "reason"),
    [
        pytest.param(
            {"example.com"},
            None,
            b"GET http://other.example/ HTTP/1.1\r\nHost: other.example\r\n\r\n",
            b"403",
            b"blocked-by-allowlist",
            id="a-host-outside-the-allowlist",
        ),
        pytest.param(
            {"pypi.org"},
            None,
            b"GET http://notpypi.org/ HTTP/1.1\r\nHost: notpypi.org\r\n\r\n",
            b"403",
            b"blocked-by-allowlist",
            id="a-host-sharing-an-allowed-suffix",
        ),
        pytest.param(
            {"pypi.org"},
            None,
            b"GET http://pypi.org.evil.com/ HTTP/1.1\r\nHost: pypi.org.evil.com\r\n\r\n",
            b"403",
            b"blocked-by-allowlist",
            id="an-allowed-name-as-a-prefix",
        ),
        pytest.param(
            {ANY_HOST},
            None,
            b"GET / HTTP/1.1\r\n\r\n",
            b"403",
            b"blocked-by-allowlist",
            id="no-host-at-all",
        ),
        pytest.param(
            {"pypi.org"},
            None,
            b"GET http://files.pypi.org/ HTTP/1.1\r\nHost: files.pypi.org\r\n\r\n",
            b"502",
            b"upstream-failure",
            id="a-subdomain-of-an-allowed-host-is-carried",
        ),
        pytest.param(
            {ANY_HOST},
            None,
            b"CONNECT 127.0.0.1:8765 HTTP/1.1\r\nHost: 127.0.0.1:8765\r\n\r\n",
            b"403",
            b"blocked-by-address",
            id="connect-to-a-substrate-address",
        ),
        pytest.param(
            {ANY_HOST},
            None,
            b"GET http://internal.example:8765/ HTTP/1.1\r\nHost: internal.example\r\n\r\n",
            b"403",
            b"blocked-by-address",
            id="a-name-resolving-into-the-substrate",
        ),
        pytest.param(
            {"example.com"},
            "secret",
            b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n\r\n",
            b"407",
            b"Proxy-Authenticate",
            id="a-visitor-proxy-without-its-credential",
        ),
        pytest.param(
            {"example.com"},
            "secret",
            b"GET http://example.com/ HTTP/1.1\r\nHost: example.com\r\n"
            b"Proxy-Authorization: Basic " + basic("another").encode() + b"\r\n\r\n",
            b"407",
            b"Proxy-Authenticate",
            id="a-visitor-proxy-with-another-visitors-credential",
        ),
    ],
)
def test_the_proxy_carries_only_what_its_allowlist_permits_and_says_why_not(
    monkeypatch: pytest.MonkeyPatch,
    allowed: set[str],
    token: str | None,
    request_bytes: bytes,
    status: bytes,
    reason: bytes,
) -> None:
    internet(monkeypatch, resolve={"internal.example": "127.0.0.1"})
    with proxy(allowed, token=token) as egress:
        response = exchange(egress.socket_path, request_bytes)

    assert response.split(b"\r\n", 1)[0].split()[1] == status
    assert reason in response


def test_the_proxy_normalizes_http_framing_in_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class Chunked(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            seen.update(
                host=self.headers.get("Host"),
                length=self.headers.get("Content-Length"),
                transfer=self.headers.get("Transfer-Encoding"),
            )
            seen["body"] = self.rfile.read(int(seen["length"] or 0))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"5\r\nhello\r\n0\r\n\r\n")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with serving(Chunked) as port:
        internet(monkeypatch, servers={"upstream.example": port})
        with proxy({"upstream.example"}) as egress:
            response = exchange(
                egress.socket_path,
                b"POST http://upstream.example/ HTTP/1.1\r\nHost: forbidden.example\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n",
            )

    headers, _, body = response.partition(b"\r\n\r\n")
    assert b"200" in headers.split(b"\r\n", 1)[0]
    assert b"transfer-encoding" not in headers.lower()
    assert body == b"hello"
    assert seen == {
        "host": "upstream.example",
        "length": "11",
        "transfer": None,
        "body": b"hello world",
    }


@pytest.mark.parametrize(
    ("header", "relayed"),
    [
        pytest.param(b"Content-Type: text/plain", True, id="an-ordinary-header"),
        pytest.param(b"X-Meta:", True, id="an-empty-value"),
        pytest.param(b"X-Evil: a\r\n b", False, id="a-folded-value"),
        pytest.param(b"X-Evil: a\x00b", False, id="a-control-character"),
    ],
)
def test_the_proxy_relays_an_upstream_header_only_as_the_upstream_sent_it(
    monkeypatch: pytest.MonkeyPatch, header: bytes, relayed: bool
) -> None:
    """An upstream header is remote text, and a folded value keeps its CRLF
    through http.client. Relayed verbatim it would carry headers of its own
    into the response the workspace reads, so a field outside the grammar
    fails the whole response rather than being quietly repaired."""
    with raw_upstream(b"HTTP/1.1 200 OK\r\n" + header + b"\r\nContent-Length: 2\r\n\r\nok") as port:
        internet(monkeypatch, servers={"upstream.example": port})
        with proxy({"upstream.example"}) as egress:
            response = exchange(
                egress.socket_path,
                b"GET http://upstream.example/ HTTP/1.1\r\nHost: upstream.example\r\n\r\n",
            )

    if relayed:
        assert response.startswith(b"HTTP/1.1 200")
        assert header.split(b":")[0] in response
    else:
        assert response.startswith(b"HTTP/1.1 502")
        assert b"X-Proxy-Error: unrelayable-upstream-header" in response


@contextmanager
def serving(handler: type[BaseHTTPRequestHandler]) -> Iterator[int]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@contextmanager
def raw_upstream(answer: bytes, listener: socket.socket | None = None) -> Iterator[int]:
    """A server that reads one request to its blank line and answers ``answer``."""
    server = listener or socket.create_server(("127.0.0.1", 0))
    server.settimeout(5)

    def serve() -> None:
        with server, server.accept()[0] as connection:
            request = b""
            while b"\r\n\r\n" not in request:
                request += connection.recv(65536)
            connection.sendall(answer)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield server.getsockname()[1] if server.family == socket.AF_INET else 0
    finally:
        thread.join(5)


@contextmanager
def half_close_upstream(listener: socket.socket) -> Iterator[list[bytes]]:
    """Reads a connection to its EOF, records it, then answers ``RESPONSE``."""
    requests: list[bytes] = []
    listener.settimeout(5)

    def serve() -> None:
        with listener, listener.accept()[0] as connection:
            requests.append(drain(connection))
            connection.sendall(RESPONSE)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield requests
    finally:
        thread.join(5)
        assert not thread.is_alive()


@contextmanager
def bridge(
    egress: Egress, cwd: Path, **options: Any
) -> Iterator[tuple[int, subprocess.Popen[bytes]]]:
    """The bridge process serving ``egress`` on a free loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
    argv, config = egress.bridge_command(port=port, **options)
    process = subprocess.Popen(
        argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(config)
        process.stdin.flush()
        assert process.stdout.readline() == b"ready\n"
        yield port, process
    finally:
        process.terminate()
        process.wait(timeout=5)


@contextmanager
def through_the_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[socket.socket, list[bytes]]]:
    del monkeypatch
    with sockets() as directory:
        egress = Egress(directory, {"example.com"})
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        upstream.bind(str(egress.socket_path))
        upstream.listen()
        with (
            half_close_upstream(upstream) as requests,
            bridge(egress, directory) as (port, _process),
            socket.create_connection(("127.0.0.1", port), timeout=5) as client,
        ):
            yield client, requests


@contextmanager
def through_a_connect_tunnel(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[socket.socket, list[bytes]]]:
    upstream = socket.create_server(("127.0.0.1", 0))
    internet(monkeypatch, servers={"example.com": upstream.getsockname()[1]})
    with (
        half_close_upstream(upstream) as requests,
        proxy({"example.com"}) as egress,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client,
    ):
        client.settimeout(5)
        client.connect(str(egress.socket_path))
        client.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
        established = b""
        while b"\r\n\r\n" not in established:
            established += client.recv(4096)
        assert b"200 Connection established" in established
        yield client, requests


@contextmanager
def through_a_peer_forward(_: pytest.MonkeyPatch) -> Iterator[tuple[socket.socket, list[bytes]]]:
    upstream = socket.create_server(("127.0.0.1", 0))
    with (
        half_close_upstream(upstream) as requests,
        proxy(set(), peers=[Peer("service", upstream.getsockname()[1])]) as egress,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client,
    ):
        ((_, _, peer_socket),) = json.loads(egress.bridge_command()[1])
        client.settimeout(5)
        client.connect(peer_socket)
        yield client, requests


@contextmanager
def through_a_connection_relay(
    _: pytest.MonkeyPatch,
) -> Iterator[tuple[socket.socket, list[bytes]]]:
    requests: list[bytes] = []

    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with serving(Upstream) as port:
        relay = ConnectionRelay(
            Connection(
                name="inference",
                capability="ssh",
                url=f"http://127.0.0.1:{port}/v1",
                headers={"Authorization": "Bearer scoped-runtime-token"},
            )
        )
        relay.start()
        try:
            with socket.create_connection(("127.0.0.1", relay.port), timeout=5) as client:
                yield client, requests
        finally:
            relay.stop()


@pytest.mark.parametrize(
    "route",
    [
        pytest.param(through_the_bridge, id="bridge"),
        pytest.param(through_a_connect_tunnel, id="connect-tunnel"),
        pytest.param(through_a_peer_forward, id="peer-forward"),
        pytest.param(through_a_connection_relay, id="connection-relay"),
    ],
)
def test_a_relay_delivers_the_whole_response_after_the_client_half_closes(
    monkeypatch: pytest.MonkeyPatch,
    route: Callable[[pytest.MonkeyPatch], Any],
) -> None:
    with route(monkeypatch) as (client, requests):
        client.settimeout(5)
        client.sendall(REQUEST)
        client.shutdown(socket.SHUT_WR)
        received = drain(client)

    assert received.endswith(BODY)
    assert requests[0].endswith(b"request body")


def test_the_bridge_ignores_workspace_imports_and_closes_reset_connections(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    package = workspace / "hud" / "environment"
    package.mkdir(parents=True)
    (workspace / "hud" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    marker = workspace / "imported"
    (package / "utils.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
    )

    with sockets() as directory:
        egress = Egress(directory, {"example.com"})
        upstream_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        upstream_listener.bind(str(egress.socket_path))
        upstream_listener.listen()
        upstream_listener.settimeout(5)
        with upstream_listener, bridge(egress, workspace) as (port, process):
            command_line = subprocess.run(
                ["ps", "-o", "command=", "-p", str(process.pid)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            client = socket.create_connection(("127.0.0.1", port), timeout=5)
            upstream, _ = upstream_listener.accept()
            with upstream:
                upstream.settimeout(1)
                client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                client.close()
                closed = upstream.recv(1)

    assert not marker.exists()
    # Where the sockets live is configuration, not something a process listing shows.
    assert str(directory) not in command_line
    assert closed == b""


VISITOR = """
import base64, http.client, os, pathlib, sys, time, urllib.parse

name, own_host, other_host = sys.argv[1:]
workspace = pathlib.Path.cwd()
(workspace / f"started-{name}").write_text(os.environ["http_proxy"])
while not (workspace / f"release-{name}").exists():
    time.sleep(0.01)

def status(host, proxy_url):
    proxy = urllib.parse.urlsplit(proxy_url)
    secret = urllib.parse.unquote(proxy.password)
    connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
    credential = base64.b64encode(f"{proxy.username}:{secret}".encode()).decode()
    authorization = {"Proxy-Authorization": f"Basic {credential}"}
    connection.request("GET", f"http://{host}/", headers=authorization)
    return str(connection.getresponse().status)

proxy_url = os.environ["http_proxy"]
statuses = [status(own_host, proxy_url), status(other_host, proxy_url)]
stale = workspace / "stale-proxy"
if stale.exists():
    statuses.append(status(own_host, stale.read_text()))
print(" ".join(statuses))
"""


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_overlapping_verifier_runs_keep_their_own_policies_and_credentials(
    tmp_path: Path,
) -> None:
    """Each verifier run is held to its own allowlist behind its own credential,
    and a second run waits for the first to release the visitor route."""
    root = tmp_path / "root"

    def visitor(name: str, own: str, other: str) -> Any:
        return workspace.run(
            ["python3", "-c", VISITOR, name, own, other], allowed_hosts={own}, identity=None
        )

    with sockets() as credentials:
        async with shell(root, require_isolation=True, credentials_dir=credentials) as (
            workspace,
            _,
        ):
            first = asyncio.create_task(visitor("first", "first.example", "second.example"))
            await eventually(lambda: (root / "started-first").exists())
            second = asyncio.create_task(visitor("second", "second.example", "first.example"))
            await asyncio.sleep(0.2)
            second_waited = not (root / "started-second").exists()
            (root / "release-first").touch()
            first_result = await first

            await eventually(lambda: (root / "started-second").exists())
            first_proxy = (root / "started-first").read_text()
            (root / "stale-proxy").write_text(first_proxy)
            (root / "release-second").touch()
            second_result = await second
            second_proxy = (root / "started-second").read_text()
            route_left_open = (credentials / "visitor" / "egress.sock").exists()

    assert second_waited
    assert first_proxy != second_proxy
    # An allowed host is carried (and fails upstream: nothing resolves in here).
    assert first_result.stdout.decode().split() == ["502", "403"]
    assert second_result.stdout.decode().split() == ["502", "403", "407"]
    assert not route_left_open


PROXIED_STATUSES = """
import base64, http.client, os, sys, urllib.parse

proxy = urllib.parse.urlsplit(os.environ["http_proxy"])
headers = {}
if proxy.username:
    credential = f"{proxy.username}:{urllib.parse.unquote(proxy.password)}"
    headers["Proxy-Authorization"] = "Basic " + base64.b64encode(credential.encode()).decode()

def status(host):
    connection = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=5)
    connection.request("GET", f"http://{host}/", headers=headers)
    return str(connection.getresponse().status)

print(" ".join(status(host) for host in sys.argv[1:]))
"""


@pytest.mark.e2e
@pytest.mark.sandbox
@pytest.mark.parametrize(
    ("options", "allowed"),
    [
        pytest.param(
            {"allowed_hosts": {"pypi.org"}}, None, id="no-policy-of-its-own-uses-the-sessions"
        ),
        pytest.param(
            {"allowed_hosts": {"pypi.org"}}, {"pypi.org"}, id="the-sessions-policy-is-its-own"
        ),
        pytest.param(
            {"peers": [Peer("db", 5432)], "ports": [5432]},
            {"pypi.org"},
            id="a-policy-of-its-own-gets-the-visitor-route",
        ),
    ],
)
async def test_a_verifier_run_is_held_to_the_policy_it_runs_under(
    tmp_path: Path, options: dict[str, Any], allowed: set[str] | None
) -> None:
    async with shell(tmp_path / "root", require_isolation=True, **options) as (workspace, _):
        result = await workspace.run(
            ["python3", "-c", PROXIED_STATUSES, "pypi.org", "example.com"],
            allowed_hosts=allowed,
            identity=None,
        )

    permitted, refused = result.stdout.decode().split()
    assert permitted != "403"
    assert refused == "403"


@contextmanager
def relay_to(upstream: type[BaseHTTPRequestHandler], **connection: Any) -> Iterator[int]:
    """A connection relay in front of ``upstream``, holding the controller's credential."""
    with serving(upstream) as port:
        relay = ConnectionRelay(
            Connection(
                name="inference",
                capability="ssh",
                url=f"http://127.0.0.1:{port}{connection.get('prefix', '')}",
                headers=connection.get("headers", {"Authorization": "Bearer scoped-runtime-token"}),
            )
        )
        relay.start()
        try:
            yield relay.port
        finally:
            relay.stop()


def test_a_connection_relay_replaces_credentials_and_preserves_streaming() -> None:
    seen: dict[str, Any] = {}

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.update(
                authorization=self.headers.get("Authorization"),
                api_key=self.headers.get("X-Api-Key"),
                body=self.rfile.read(int(self.headers.get("Content-Length", "0"))),
            )
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"5\r\nfirst\r\n6\r\nsecond\r\n0\r\n\r\n")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    token = "scoped-runtime-token"
    headers = {"Authorization": f"Bearer {token}", "X-Api-Key": token}
    with relay_to(Upstream, prefix="/v1", headers=headers) as port:
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        client.request(
            "POST",
            "/v1/messages",
            body=b"request",
            headers={"Authorization": "Bearer model-visible", "X-Api-Key": "model-visible"},
        )
        response = client.getresponse()
        status, body = response.status, response.read()
        client.close()

    assert (status, body) == (200, b"firstsecond")
    assert seen == {"authorization": f"Bearer {token}", "api_key": token, "body": b"request"}


def test_a_connection_relay_refuses_paths_that_could_leave_its_prefix() -> None:
    forwarded: list[str] = []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            forwarded.append(self.path)
            self.send_response(204)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    answers: dict[str, tuple[int, str | None]] = {}
    with relay_to(Upstream, prefix="/v1") as port:
        for path in (
            "/v1/../admin",
            "/v1/./models",
            "/v1/%2e%2e/admin",
            "/v1/%2E%2E/admin",
            "/v1/..%2fadmin",
            "/v1/..%5cadmin",
            "/v1/models",
        ):
            client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            client.request("GET", path)
            response = client.getresponse()
            response.read()
            answers[path] = (response.status, response.getheader("X-Connection-Error"))
            client.close()

    assert forwarded == ["/v1/models"]
    assert answers.pop("/v1/models") == (204, None)
    assert set(answers.values()) == {(400, "non-canonical-path")}


@pytest.mark.parametrize("placeholder", ["X-Api-Key", "Api-Key", "X-Goog-Api-Key"])
def test_a_connection_relay_drops_process_credentials_the_connection_does_not_set(
    placeholder: str,
) -> None:
    received: dict[str, str] = {}

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.update({key.casefold(): value for key, value in self.headers.items()})
            self.send_response(204)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with relay_to(Upstream) as port:
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        client.request("POST", "/v1/messages", body=b"{}", headers={placeholder: "model-visible"})
        status = client.getresponse().status
        client.close()

    assert status == 204
    assert received["authorization"] == "Bearer scoped-runtime-token"
    assert placeholder.casefold() not in received
    assert "model-visible" not in received.values()
