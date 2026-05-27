"""Agent ABC."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from hud.capabilities import CapabilityClient
    from hud.client import Manifest

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Minimal agent contract.

    * ``initialize(manifest)`` — open clients for every supported binding.
    * ``run(...)`` — subclass-defined.
    * ``close()`` — release opened clients.
    """

    clients: ClassVar[tuple[type[CapabilityClient], ...]] = ()
    connections: dict[str, CapabilityClient]

    async def initialize(self, manifest: Manifest) -> None:
        by_protocol = {cls.protocol: cls for cls in type(self).clients}
        pairs = [
            (b, by_protocol[b.protocol]) for b in manifest.bindings if b.protocol in by_protocol
        ]
        opened = await asyncio.gather(*(cls.connect(b) for b, cls in pairs))
        self.connections = {b.name: c for (b, _), c in zip(pairs, opened, strict=False)}

    async def close(self) -> None:
        for client in getattr(self, "connections", {}).values():
            with contextlib.suppress(Exception):
                await client.close()
        self.connections = {}
