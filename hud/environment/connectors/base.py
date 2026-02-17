"""Base connector mixin with shared helper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp.tools.tool import Tool

    from hud.environment.connection import ConnectionType, Connector

__all__ = ["BaseConnectorMixin"]


class BaseConnectorMixin:
    """Base mixin providing connection helper.

    Requires:
        _connections: dict[str, Connector]
    """

    _connections: dict[str, Connector]

    def _add_connection(
        self,
        name: str,
        transport: Any,
        *,
        connection_type: ConnectionType,
        auth: str | None = None,
        prefix: str | None = None,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        transform: Callable[[Tool], Tool | None] | None = None,
    ) -> Any:
        """Add a connection to the environment.

        Args:
            name: Connection name/alias.
            transport: FastMCP transport (URL, config dict, etc.).
            connection_type: LOCAL or REMOTE - determines parallelization.
            auth: Authorization header value.
            prefix: Prefix for tool names.
            include: Only include these tools.
            exclude: Exclude these tools.
            transform: Transform function for tools.

        Returns:
            self for chaining.
        """
        from hud.environment.connection import ConnectionConfig, Connector

        config = ConnectionConfig(
            prefix=prefix,
            include=include,
            exclude=exclude,
            transform=transform,
        )
        self._connections[name] = Connector(
            transport,
            config,
            name,
            connection_type=connection_type,
            auth=auth,
        )
        return self
