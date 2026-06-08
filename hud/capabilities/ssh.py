"""SSHClient — asyncssh connection wrapper."""

from __future__ import annotations

from typing import ClassVar, Self
from urllib.parse import urlsplit

import asyncssh

from .base import Capability, CapabilityClient


def _known_hosts_host(hostname: str, port: int) -> str:
    """Format a known_hosts host token (bracketed ``[host]:port`` for non-22 ports)."""
    return hostname if port == 22 else f"[{hostname}]:{port}"


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
        host_pubkey = cap.params.get("host_pubkey")
        if not host_pubkey:
            # Fail closed: the capability publishes the daemon's host key precisely
            # so we can pin it. Without it we would be open to MITM, so refuse
            # rather than silently falling back to known_hosts=None.
            raise ValueError(
                f"ssh capability {cap.name!r} is missing host_pubkey; refusing to "
                "connect without host-key verification.",
            )
        known_hosts = asyncssh.import_known_hosts(
            f"{_known_hosts_host(parts.hostname, parts.port)} {host_pubkey}\n",
        )
        client_key_path = cap.params.get("client_key_path")
        conn = await asyncssh.connect(
            host=parts.hostname,
            port=parts.port,
            username=cap.params.get("user", "agent"),
            client_keys=[client_key_path] if client_key_path else None,
            known_hosts=known_hosts,
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
