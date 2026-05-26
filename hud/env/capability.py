"""Capability — declarative wire metadata for one slice of env access.

A ``Capability`` is just a tuple of ``(name, protocol, endpoint)``. No
inheritance, no lifecycle. Standing up the daemon (SSH server, Chromium,
VNC server, rosbridge_server, MCP server) is the env-author's job — usually
they already run that infra. The capability just tells the harness *where*
to reach it and what's needed to authenticate.

Guiding principles:

* **Manifest = what you need to open the connection; the connection itself
  tells you everything else.** MCP has ``tools/list``, ROS 2 has
  ``rosapi/topics`` and the ``/robot_description`` topic, CDP has
  ``Target.getTargets``, RFB sends pixel dimensions in ``ServerInit``. We
  don't duplicate any of that in the manifest.
* **All endpoints are network URLs with a scheme.** No stdio, no local
  pipes — a capability is something a remote harness reaches over the
  network. The URL scheme tells you the transport (``ssh://``, ``ws://``,
  ``wss://``, ``http://``, ``https://``, ``tcp://``, ``rfb://``).

Use the well-known classmethods for catalogued protocols::

    Capability.ssh(url="ssh://127.0.0.1:2222", host_pubkey=..., client_key_path=...)
    Capability.cdp(url="ws://127.0.0.1:9222")
    Capability.rfb(url="rfb://127.0.0.1:5900")
    Capability.mcp(url="ws://127.0.0.1:9990/mcp")
    Capability.ros2(url="ws://127.0.0.1:9090")

For anything else (custom protocols, extra hint params), construct
``Capability(name, protocol, Endpoint(url=..., params=...))`` directly.

Daemon lifecycle is owned by the env-author. For the convenience case where
they want the SDK to spin up an SSH server bound to a bwrap'd workspace,
see ``Workspace`` and ``Workspace.ssh_capability()``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .utils import SCHEME_RE, normalize_url

# ─────────────────────────── core types ───────────────────────────


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Where a harness reaches a capability.

    ``url`` always carries a scheme — it's the transport indicator and the
    address all in one. ``params`` carries protocol-specific info needed at
    connection time (auth keys, tokens, etc.).
    """

    url: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Capability:
    """One wire-accessible slice of env: a ``(name, protocol, endpoint)`` tuple."""

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

    # ─────────────── well-known protocol factories ───────────────

    @classmethod
    def ssh(
        cls,
        *,
        name: str = "shell",
        url: str,                                  # "ssh://host:port" or "host:port"
        user: str = "agent",
        host_pubkey: str,
        client_key_path: str | os.PathLike[str] | None = None,
    ) -> Capability:
        """``ssh/2`` — points at an SSH daemon.

        For the SDK-managed case (bwrap-isolated shell + SFTP chroot), the
        env-author starts a ``Workspace`` and constructs this capability
        from ``workspace.ssh_url`` / ``workspace.ssh_host_pubkey`` /
        ``workspace.ssh_client_key_path``.
        """
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
        url: str,                                  # "ws://host:port[/path]" or "host:port"
        target_id: str | None = None,
    ) -> Capability:
        """``cdp/1.3`` — points at a Chromium DevTools WebSocket.

        Env-author runs Chromium with ``--remote-debugging-port=9222``.
        Targets (tabs / iframes / workers) are discovered after connect via
        ``Target.getTargets``.
        """
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
        url: str,                                  # "rfb://host:port" or "host:port"
        password: str | None = None,
    ) -> Capability:
        """``rfb/3.8`` — points at a VNC/RFB server (Xvnc, x11vnc, vncserver).

        Pixel dimensions arrive in the RFB ``ServerInit`` message after the
        handshake — not pre-published here.
        """
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
        url: str,                                  # "ws://", "wss://", "http(s)://.../sse"
        auth_token: str | None = None,
    ) -> Capability:
        """``mcp/2025-11-25`` — points at an MCP server (FastMCP, others).

        Network transports only: WebSocket or HTTP+SSE. Stdio is intentionally
        unsupported (a capability has to be reachable over the network).
        Tools are discovered via ``tools/list`` after connect.
        """
        # Reject unsupported schemes early (e.g. "stdio:cmd") before URL
        # normalization mistakes the lone scheme for a hostname.
        m = SCHEME_RE.match(url)
        if m and "://" not in url:
            scheme = m.group(1)
            raise ValueError(
                f"mcp/2025-11-25: only ws/wss/http/https URLs are supported, got {scheme!r}",
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
    def ros2(
        cls,
        *,
        name: str = "ros",
        url: str,                                  # "ws://host:9090" (rosbridge)
    ) -> Capability:
        """``ros2/2`` — points at a rosbridge-compatible WebSocket.

        Env-author runs ``rosbridge_server`` (full ROS 2) or a pure-Python
        equivalent. URDF is discovered by subscribing to ``/robot_description``
        (transient-local QoS). Topics / services / actions are discovered via
        ``rosapi/topics``, ``rosapi/services``, ``rosapi/action_servers``.
        """
        normalized = normalize_url(url, default_scheme="ws", default_port=9090)
        return cls(name=name, protocol="ros2/2", endpoint=Endpoint(normalized, {}))


__all__ = [
    "Capability",
    "Endpoint",
]
