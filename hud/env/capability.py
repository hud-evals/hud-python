"""Capability: declarative ``(name, protocol, endpoint)`` metadata.

Env-author runs the daemon (SSH/Chrome/VNC/MCP/rosbridge); capability just
publishes its URL + connection-time auth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .utils import SCHEME_RE, normalize_url


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A capability URL + connection-time params (auth keys, tokens)."""

    url: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Capability:
    """One wire-accessible slice of env."""

    name: str
    protocol: str
    endpoint: Endpoint

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "endpoint": {"url": self.endpoint.url},
            "params": dict(self.endpoint.params),
        }

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
    ) -> Capability:
        """``ssh/2`` — SSH daemon with publickey auth."""
        normalized = normalize_url(url, default_scheme="ssh", default_port=22)
        params: dict[str, Any] = {"user": user, "host_pubkey": host_pubkey}
        if client_key_path is not None:
            params["client_key_path"] = os.fspath(client_key_path)
        return cls(name=name, protocol="ssh/2", endpoint=Endpoint(normalized, params))

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
        return cls(name=name, protocol="cdp/1.3", endpoint=Endpoint(normalized, params))

    @classmethod
    def rfb(
        cls,
        *,
        name: str = "screen",
        url: str,
        password: str | None = None,
    ) -> Capability:
        """``rfb/3.8`` — VNC/RFB pixel + HID server."""
        normalized = normalize_url(url, default_scheme="rfb", default_port=5900)
        params: dict[str, Any] = {}
        if password is not None:
            params["password"] = password
        return cls(name=name, protocol="rfb/3.8", endpoint=Endpoint(normalized, params))

    @classmethod
    def mcp(
        cls,
        *,
        name: str = "tools",
        url: str,
        auth_token: str | None = None,
    ) -> Capability:
        """``mcp/2025-11-25`` — MCP server (ws/wss/http/https; no stdio)."""
        # Reject schemes like "stdio:cmd" before normalize_url mistakes the
        # scheme for a hostname.
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
        return cls(name=name, protocol="mcp/2025-11-25", endpoint=Endpoint(normalized, params))

    @classmethod
    def ros2(cls, *, name: str = "ros", url: str) -> Capability:
        """``ros2/2`` — rosbridge-compatible WebSocket."""
        normalized = normalize_url(url, default_scheme="ws", default_port=9090)
        return cls(name=name, protocol="ros2/2", endpoint=Endpoint(normalized, {}))


__all__ = ["Capability", "Endpoint"]
