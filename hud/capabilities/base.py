"""Capability declaration + CapabilityClient ABC."""

from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self
from urllib.parse import urlsplit

#: Matches the scheme prefix of a URL (RFC 3986).
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
        userinfo = f"{parts.username}@" if parts.username else ""
        path = parts.path
        query = f"?{parts.query}" if parts.query else ""
        fragment = f"#{parts.fragment}" if parts.fragment else ""
        return f"{parts.scheme}://{userinfo}{parts.hostname}:{default_port}{path}{query}{fragment}"
    return s


@dataclass(frozen=True, slots=True)
class Capability:
    """``(name, protocol, url, params)`` — declarative metadata for one slice of env access.

    Concrete declarations carry the URL of a daemon the env author runs
    (``Capability.cdp(url=...)``, ``Capability.ssh(url=...)``). A declaration
    with an **empty url** is *backed*: the env runs the daemon and resolves
    the address when it serves a client (``Capability.shell(root)`` → a
    managed ``Workspace``).
    """

    name: str
    protocol: str
    url: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "url": self.url,
            "params": dict(self.params),
        }

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> Capability:
        return cls(
            name=data["name"],
            protocol=data["protocol"],
            url=data["url"],
            params=dict(data.get("params") or {}),
        )

    # ─── well-known protocol factories ─────────────────────────────────

    @classmethod
    def shell(
        cls,
        root: str | os.PathLike[str],
        *,
        name: str = "shell",
        network: bool = False,
        guest_path: str = "/workspace",
        user: str = "agent",
    ) -> Capability:
        """``ssh/2``, backed — the env serves a managed ``Workspace`` for it.

        Declares *intent* (a shell rooted at ``root``), not an address: nothing
        is generated or bound until the env answers a client's ``hello``. For
        an SSH daemon you run yourself, declare :meth:`ssh` with its URL.
        """
        params: dict[str, Any] = {
            "root": os.fspath(root),
            "network": network,
            "guest_path": guest_path,
            "user": user,
        }
        return cls(name=name, protocol="ssh/2", url="", params=params)

    @classmethod
    def ssh(
        cls,
        *,
        name: str = "shell",
        url: str,
        user: str = "agent",
        host_pubkey: str,
        client_key_path: str | os.PathLike[str] | None = None,
        shell: str | None = None,
    ) -> Capability:
        """``ssh/2`` — SSH daemon with publickey auth.

        ``shell`` declares the remote shell type (``bash``, ``powershell``,
        ``cmd``). Defaults to auto-detect from ``sys.platform`` at
        construction time. Agents read this to format commands correctly.
        """
        normalized = normalize_url(url, default_scheme="ssh", default_port=22)
        if shell is None:
            shell = "cmd" if sys.platform == "win32" else "bash"
        params: dict[str, Any] = {"user": user, "host_pubkey": host_pubkey, "shell": shell}
        if client_key_path is not None:
            params["client_key_path"] = os.fspath(client_key_path)
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

        ``display`` selects the VNC display number (standard convention: display
        ``N`` listens on port ``5900 + N``). When the URL omits an explicit port
        the port defaults to ``5900 + display``; an explicit port in the URL
        always wins. Envs hosting multiple screens publish one rfb capability
        per display, e.g.::

            Capability.rfb(name="screen-0", url="rfb://host", display=0)
            Capability.rfb(name="screen-1", url="rfb://host", display=1)
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
        params: dict[str, Any] = {}
        if auth_token is not None:
            params["auth_token"] = auth_token
        return cls(name=name, protocol="mcp/2025-11-25", url=normalized, params=params)

    @classmethod
    def ros2(cls, *, name: str = "ros", url: str) -> Capability:
        """``ros2/2`` — rosbridge-compatible WebSocket."""
        normalized = normalize_url(url, default_scheme="ws", default_port=9090)
        return cls(name=name, protocol="ros2/2", url=normalized, params={})


class CapabilityClient(ABC):
    """Live connection to a Capability. Subclasses expose protocol-native methods."""

    protocol: ClassVar[str]

    @classmethod
    @abstractmethod
    async def connect(cls, cap: Capability) -> Self: ...

    @abstractmethod
    async def close(self) -> None: ...


__all__ = ["Capability", "CapabilityClient"]
