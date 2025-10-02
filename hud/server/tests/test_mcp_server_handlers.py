from __future__ import annotations

from typing import Any, cast

from hud.server import MCPServer
from hud.server.low_level import LowLevelServerWithInit


def test_notification_handlers_preserved_on_replacement():
    """When init server replaces low-level server, notification handlers must be kept."""
    mcp = MCPServer(name="PreserveNotif")

    # Seed a fake notification handler on the pre-replacement server
    before = mcp._mcp_server
    cast("dict[Any, Any]", before.notification_handlers)["foo/notify"] = object()

    @mcp.initialize
    async def _init(_ctx) -> None:
        pass

    after = mcp._mcp_server
    assert isinstance(after, LowLevelServerWithInit)
    assert after is not before, "low-level server should be replaced once"
    # Must still contain our seeded handler (dict is copied over)
    assert "foo/notify" in after.notification_handlers


def test_init_server_replacement_is_idempotent():
    """Second @initialize must NOT replace the low-level server again."""
    mcp = MCPServer(name="InitIdempotent")

    @mcp.initialize
    async def _a(_ctx) -> None:
        pass

    first = mcp._mcp_server

    @mcp.initialize
    async def _b(_ctx) -> None:
        # last initializer should win, but server object should not be replaced again
        pass

    second = mcp._mcp_server
    assert first is second, "Server replacement should occur at most once per instance"
