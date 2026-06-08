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
    """``(name, protocol, url, params)`` — declarative wire metadata for one slice of env access.

    Env-author runs the daemon; capability publishes the URL + connection-time auth.
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
        if not isinstance(data, dict):
            raise TypeError("capability manifest must be an object")
        name = data.get("name")
        protocol = data.get("protocol")
        url = data.get("url")
        params = data.get("params", {})
        if not isinstance(name, str) or not name:
            raise ValueError("capability manifest requires non-empty string 'name'")
        if not isinstance(protocol, str) or not protocol:
            raise ValueError("capability manifest requires non-empty string 'protocol'")
        if not isinstance(url, str) or not url:
            raise ValueError("capability manifest requires non-empty string 'url'")
        if not isinstance(params, dict):
            raise TypeError("capability manifest 'params' must be an object")

        cls._validate_manifest_params(protocol, params)
        return cls(
            name=name,
            protocol=protocol,
            url=url,
            params=dict(params),
        )

    @staticmethod
    def _validate_manifest_params(protocol: str, params: dict[str, Any]) -> None:
        family = protocol.split("/", 1)[0]
        if family == "ssh":
            for key in ("user", "host_pubkey", "client_key_path", "shell"):
                if key in params and not isinstance(params[key], str):
                    raise TypeError(f"ssh capability param {key!r} must be a string")
            return
        if family == "cdp":
            if "target_id" in params and not isinstance(params["target_id"], str):
                raise TypeError("cdp capability param 'target_id' must be a string")
            return
        if family == "rfb":
            display = params.get("display")
            if display is not None and (not isinstance(display, int) or isinstance(display, bool)):
                raise TypeError("rfb capability param 'display' must be an integer")
            if "password" in params and not isinstance(params["password"], str):
                raise TypeError("rfb capability param 'password' must be a string")
            return
        if family == "mcp" and "auth_token" in params and not isinstance(params["auth_token"], str):
            raise TypeError("mcp capability param 'auth_token' must be a string")

    # ─── well-known protocol factories ─────────────────────────────────

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
        always wins.

        Note: the client currently resolves one binding per protocol, so a single
        ``rfb`` capability is supported per env (multi-screen is not yet wired).
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
