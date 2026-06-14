"""Deprecated shim: ``hud.server.MCPServer`` is now ``fastmcp.FastMCP``.

The HUD ``MCPServer`` wrapper was removed in v6 — custom MCP tools run on a
plain FastMCP server. This module keeps ``from hud.server import MCPServer``
importable (aliased to :class:`fastmcp.FastMCP`) and emits a
``DeprecationWarning``, so an existing env keeps serving while you migrate.

The common surface is unchanged: ``MCPServer(name=...)``, ``@server.tool``,
and ``server.run_async(transport="http", host=..., port=...)`` all work on
``FastMCP``. Wrapper-only extras (server-side ``initialize``/``shutdown``
hooks, SIGTERM handling) are gone — drive the lifecycle with
``@env.initialize`` / ``@env.shutdown`` on your :class:`~hud.environment.Environment`.
"""

from __future__ import annotations

import warnings

from fastmcp import FastMCP

warnings.warn(
    "hud.server.MCPServer was removed in v6: use `from fastmcp import FastMCP` "
    "directly (same `@server.tool` and `run_async`), and manage its lifecycle "
    "with @env.initialize / @env.shutdown on your Environment.",
    DeprecationWarning,
    stacklevel=2,
)

#: Back-compat alias. New code should import ``FastMCP`` from ``fastmcp``.
MCPServer = FastMCP

__all__ = ["MCPServer"]
