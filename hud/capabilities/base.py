"""Capability declarations + CapabilityClient ABC."""

from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Iterable

# Matches the scheme prefix of a URL (RFC 3986).
SCHEME_RE: re.Pattern[str] = re.compile(r"^([a-zA-Z][a-zA-Z0-9+\-.]*):")


def normalize_url(url: str, *, default_scheme: str, default_port: int | None) -> str:
    """Coerce shorthand ``host[:port]`` into a full ``scheme://host:port[/path]`` URL."""
    s = url if "://" in url else f"{default_scheme}://{url}"
    parts = urlsplit(s)
    if parts.scheme == "":
        raise ValueError(f"invalid URL (no scheme): {url!r}")
    if parts.hostname is None:
        raise ValueError(f"invalid URL (no host): {url!r}")
    if parts.port is None and default_port is not None:
        userinfo = f"{parts.netloc.rpartition('@')[0]}@" if "@" in parts.netloc else ""
        hostname = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        return urlunsplit(
            (
                parts.scheme,
                f"{userinfo}{hostname}:{default_port}",
                parts.path,
                parts.query,
                parts.fragment,
            )
        )
    return s


@dataclass(frozen=True, slots=True)
class Capability:
    """``(name, protocol, url, params)`` — concrete wire data for one slice of env access.

    ``url`` is the address a ``CapabilityClient`` dials. ``env_url`` is where the environment
    serves it, which code running inside the environment dials; the two differ once a
    session routes the capability to this process.
    """

    name: str
    protocol: str
    url: str
    params: dict[str, Any] = field(default_factory=dict)
    env_url: str = ""

    def __post_init__(self) -> None:
        if not self.env_url:
            object.__setattr__(self, "env_url", self.url)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "url": self.url,
            "params": dict(self.params),
        }

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Capability:
        protocol = data["protocol"]
        url = data["url"]
        params = dict(data["params"]) if "params" in data and data["params"] is not None else {}
        if protocol.split("/", 1)[0] == "mcp" and "transport" not in params:
            params["transport"] = (
                "websocket" if urlsplit(url).scheme in {"ws", "wss"} else "streamable-http"
            )
        return cls(
            name=data["name"],
            protocol=protocol,
            url=url,
            params=params,
        )

    @classmethod
    def ssh(
        cls,
        *,
        name: str = "shell",
        url: str,
        user: str = "agent",
        host_pubkey: str,
        client_key: str | None = None,
        client_key_path: str | os.PathLike[str] | None = None,
        shell: str | None = None,
        cwd: str | None = None,
        isolation: Literal["bwrap", "none"] | None = None,
        process_connections: bool = False,
    ) -> Capability:
        """``ssh/2`` — SSH daemon with publickey auth.

        ``client_key`` is the private key's content and works across network
        namespaces; ``client_key_path`` works only when client and daemon share a
        filesystem. ``shell`` (``bash``, ``powershell`` or ``cmd``) defaults to this
        platform's. ``cwd`` is where sessions start, a path in the session's own
        namespace. ``isolation`` records the sandbox actually granted to the workspace.
        """
        normalized = normalize_url(url, default_scheme="ssh", default_port=22)
        if shell is None:
            shell = "cmd" if sys.platform == "win32" else "bash"
        params: dict[str, Any] = {"user": user, "host_pubkey": host_pubkey, "shell": shell}
        if client_key is not None:
            params["client_key"] = client_key
        if client_key_path is not None:
            params["client_key_path"] = os.fspath(client_key_path)
        if cwd is not None:
            params["cwd"] = cwd
        if isolation is not None:
            params["isolation"] = isolation
        if process_connections:
            params["process_connections"] = True
        return cls(name=name, protocol="ssh/2", url=normalized, params=params)

    @classmethod
    def cdp(
        cls,
        *,
        name: str = "browser",
        url: str,
        target_id: str | None = None,
    ) -> Capability:
        """``cdp/1.3`` — Chromium DevTools over WebSocket."""
        normalized = normalize_url(url, default_scheme="ws", default_port=9222)
        params: dict[str, Any] = {}
        if target_id is not None:
            params["target_id"] = target_id
        return cls(name=name, protocol="cdp/1.3", url=normalized, params=params)

    @classmethod
    def rfb(
        cls,
        *,
        name: str = "screen",
        url: str,
        password: str | None = None,
        display: int = 0,
    ) -> Capability:
        """``rfb/3.8`` — VNC/RFB pixel + HID server.

        A URL without a port gets ``5900 + display``. Envs with several screens publish
        one capability per display.
        """
        normalized = normalize_url(url, default_scheme="rfb", default_port=5900 + display)
        params: dict[str, Any] = {"display": display}
        if password is not None:
            params["password"] = password
        return cls(name=name, protocol="rfb/3.8", url=normalized, params=params)

    @classmethod
    def mcp(
        cls,
        *,
        name: str = "tools",
        url: str,
        auth_token: str | None = None,
        transport: Literal["sse", "streamable-http", "websocket"] | None = None,
    ) -> Capability:
        """``mcp/2025-11-25`` — MCP server (ws/wss/http/https; no stdio)."""
        m = SCHEME_RE.match(url)
        if m and "://" not in url:
            raise ValueError(
                f"mcp/2025-11-25: only ws/wss/http/https URLs are supported, got {m.group(1)!r}",
            )
        normalized = normalize_url(url, default_scheme="ws", default_port=None)
        scheme = urlsplit(normalized).scheme
        if scheme not in {"ws", "wss", "http", "https"}:
            raise ValueError(
                f"mcp/2025-11-25: only ws/wss/http/https URLs are supported, got {scheme!r}",
            )
        if transport == "websocket" and scheme not in {"ws", "wss"}:
            raise ValueError("mcp websocket transport requires a ws:// or wss:// URL")
        if transport in {"sse", "streamable-http"} and scheme not in {"http", "https"}:
            raise ValueError(f"mcp {transport} transport requires an http:// or https:// URL")
        if transport is None:
            transport = "websocket" if scheme in {"ws", "wss"} else "streamable-http"
        normalized = normalize_url(
            normalized,
            default_scheme="ws",
            default_port=443 if scheme in {"https", "wss"} else 80,
        )
        params: dict[str, Any] = {"transport": transport}
        if auth_token is not None:
            params["auth_token"] = auth_token
        return cls(name=name, protocol="mcp/2025-11-25", url=normalized, params=params)

    @classmethod
    def robot(
        cls,
        *,
        name: str = "robot",
        url: str,
        contract: dict[str, Any],
    ) -> Capability:
        """``openpi/0`` — schema-driven action/observation loop over WebSocket.

        ``contract`` is the env's self-describing config, passed verbatim through the
        manifest: ``robot_type``, ``control_rate``, and ``features``, each declaring
        its ``role`` (``"action"`` or ``"observation"``), layout and normalization
        ``stats``.
        """
        normalized = normalize_url(url, default_scheme="ws", default_port=9091)
        return cls(name=name, protocol="openpi/0", url=normalized, params={"contract": contract})


def find_capability(capabilities: Iterable[Capability], ref: str) -> Capability:
    """The one capability ``ref`` names: its name, protocol, or protocol family."""
    capabilities = list(capabilities)
    matches = [
        capability
        for capability in capabilities
        if ref in (capability.name, capability.protocol, capability.protocol.split("/", 1)[0])
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        names = ", ".join(f"{c.name} ({c.protocol})" for c in matches)
        raise KeyError(f"ambiguous capability {ref!r}; matches: {names}")
    available = ", ".join(f"{c.name} ({c.protocol})" for c in capabilities)
    raise KeyError(f"no capability {ref!r} (available: {available or '<none>'})")


class CapabilityClient(ABC):
    """Live connection to a Capability. Subclasses expose protocol-native methods."""

    protocol: ClassVar[str]

    @classmethod
    @abstractmethod
    async def connect(cls, cap: Capability) -> Self: ...

    @abstractmethod
    async def close(self) -> None: ...
