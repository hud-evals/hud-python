"""Typer groups that own command dispatch."""

from __future__ import annotations

from typing import Any

from typer.core import TyperGroup


class ImplicitGetGroup(TyperGroup):
    """Treat a bare id as ``get <id>`` (``hud jobs <id>``, ``hud trace <id>``)."""

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args.insert(0, "get")
        return super().resolve_command(ctx, args)
