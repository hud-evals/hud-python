"""SSHClient — asyncssh connection wrapper."""

from __future__ import annotations

from typing import ClassVar, Self
from urllib.parse import urlsplit

import asyncssh

from .base import Capability, CapabilityClient


class SSHClient(CapabilityClient):
    """Thin asyncssh wrapper. Exposes the raw connection via ``conn``."""

    protocol: ClassVar[str] = "ssh/2"

    def __init__(self, capability: Capability, conn: asyncssh.SSHClientConnection) -> None:
        self.capability = capability
        self._conn = conn

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        parts = urlsplit(cap.url)
        if parts.hostname is None or parts.port is None:
            raise ValueError(f"ssh capability missing host or port: {cap.url!r}")
        client_key_path = cap.params.get("client_key_path")
        conn = await asyncssh.connect(
            host=parts.hostname,
            port=parts.port,
            username=cap.params.get("user", "agent"),
            client_keys=[client_key_path] if client_key_path else None,
            known_hosts=None,
        )
        return cls(cap, conn)

    @property
    def conn(self) -> asyncssh.SSHClientConnection:
        """Raw asyncssh connection — use for ``run``, SFTP, port forwarding, etc."""
        return self._conn

    async def close(self) -> None:
        self._conn.close()
        await self._conn.wait_closed()


__all__ = ["SSHClient"]
