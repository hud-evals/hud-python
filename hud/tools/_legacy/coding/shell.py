"""Legacy shell import path."""

from __future__ import annotations

from hud.tools.coding import BashSession, BashTool


class ShellTool(BashTool):
    """Backward-compatible import name for BashTool."""

    def __init__(self, session: BashSession | None = None, cwd: str | None = None) -> None:
        super().__init__(
            session=session or (BashSession(cwd=cwd) if cwd is not None else None),
            name="bash",
            title="Bash Shell",
            description="Execute shell commands in a persistent bash session",
        )


__all__ = ["BashSession", "ShellTool"]
