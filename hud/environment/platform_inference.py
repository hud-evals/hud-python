"""Platform inference made available inside bounded workspaces."""

from __future__ import annotations

import hmac
import http.client
import secrets
import threading
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from .egress import _HOP_BY_HOP, _field, _Unrelayable

if TYPE_CHECKING:
    from collections.abc import Mapping


_CREDENTIAL_HEADERS = frozenset(
    {"authorization", "hud-api-key", "hud-runtime-token", "x-api-key", "x-goog-api-key"}
)
_TRACE_HEADERS = frozenset({"trace-id", "x-trace-id", "x-hud-trace-id"})


@dataclass(frozen=True, slots=True)
class InferenceBinding:
    """Workspace-local connection details for one platform inference lease."""

    base_url: str
    api_key: str


@dataclass(frozen=True, slots=True)
class _Lease:
    upstream: urllib.parse.SplitResult
    upstream_token: str
    trace_id: str | None
    client_key: str


def _request_key(headers: Mapping[str, str]) -> str | None:
    for name in ("hud-api-key", "x-api-key", "x-goog-api-key"):
        if value := headers.get(name):
            return value
    authorization = headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    return value if scheme.lower() == "bearer" and value else None


class _InferenceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    proxy: PlatformInferenceProxy

    def log_message(self, format: str, *args: Any) -> None:
        """Requests and opaque lease paths must not enter environment logs."""

    def _fail(self, status: int, reason: str) -> None:
        self.send_response(status)
        self.send_header("X-Proxy-Error", reason)
        self.send_header("Content-Length", "0")
        self.end_headers()

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
        parts = urllib.parse.urlsplit(self.path)
        lease, upstream_path = self.proxy.resolve(parts.path)
        if lease is None:
            self._fail(404, "unknown-lease")
            return
        supplied = _request_key({key.lower(): value for key, value in self.headers.items()})
        if supplied is None or not hmac.compare_digest(supplied, lease.client_key):
            self._fail(401, "invalid-lease-key")
            return
        try:
            body = self._request_body()
        except (ValueError, OverflowError):
            self.close_connection = True
            self._fail(400, "invalid-request-body")
            return

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_BY_HOP | _CREDENTIAL_HEADERS | _TRACE_HEADERS | {"host"}
        }
        headers["Hud-Runtime-Token"] = lease.upstream_token
        if lease.trace_id is not None:
            headers["Trace-Id"] = lease.trace_id

        upstream = lease.upstream
        base = upstream.path.rstrip("/")
        path = f"{base}/{upstream_path.lstrip('/')}"
        if parts.query:
            path = f"{path}?{parts.query}"
        host = upstream.hostname
        assert host is not None
        connection: http.client.HTTPConnection
        if upstream.scheme == "https":
            connection = http.client.HTTPSConnection(host, upstream.port, timeout=300)
        else:
            connection = http.client.HTTPConnection(host, upstream.port, timeout=300)
        response_started = False
        try:
            connection.request(self.command, path, body=body, headers=headers)
            response = connection.getresponse()
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
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            while chunk := response.read(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except _Unrelayable:
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


class PlatformInferenceProxy:
    """Environment-owned reverse proxy with per-control-session credentials."""

    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._leases: dict[str, tuple[str, _Lease]] = {}
        self._lock = threading.Lock()

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("platform inference proxy is not started")
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._server is not None:
            return
        handler = type("_ScopedInferenceHandler", (_InferenceHandler,), {"proxy": self})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()

    def register(
        self,
        session_id: str,
        *,
        upstream_url: str,
        token: str,
        trace_id: str | None,
        workspace_url: str,
    ) -> InferenceBinding:
        upstream = urllib.parse.urlsplit(upstream_url)
        if (
            upstream.scheme not in {"http", "https"}
            or upstream.hostname is None
            or upstream.username is not None
            or upstream.password is not None
            or upstream.query
            or upstream.fragment
        ):
            raise ValueError("platform inference upstream must be an HTTP(S) base URL")
        if not token:
            raise ValueError("platform inference token must not be empty")
        with self._lock:
            existing = self._leases.get(session_id)
            if existing is not None:
                route, lease = existing
                requested = (upstream, token, trace_id)
                current = (lease.upstream, lease.upstream_token, lease.trace_id)
                if requested != current:
                    raise RuntimeError("platform inference is already bound for this session")
                return InferenceBinding(f"{workspace_url}/{route}", lease.client_key)
            route = "p/" + secrets.token_urlsafe(18)
            lease = _Lease(
                upstream=upstream,
                upstream_token=token,
                trace_id=trace_id,
                client_key=secrets.token_urlsafe(32),
            )
            self._leases[session_id] = (route, lease)
        return InferenceBinding(f"{workspace_url}/{route}", lease.client_key)

    def resolve(self, path: str) -> tuple[_Lease | None, str]:
        stripped = path.lstrip("/")
        prefix, separator, remainder = stripped.partition("/")
        if prefix != "p" or not separator:
            return None, ""
        token, separator, upstream_path = remainder.partition("/")
        if not token or not separator:
            return None, ""
        route = f"p/{token}"
        with self._lock:
            for stored_route, lease in self._leases.values():
                if hmac.compare_digest(route, stored_route):
                    return lease, "/" + upstream_path
        return None, ""

    def unregister(self, session_id: str) -> None:
        with self._lock:
            self._leases.pop(session_id, None)

    def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)
        with self._lock:
            self._leases.clear()


__all__ = ["InferenceBinding", "PlatformInferenceProxy"]
