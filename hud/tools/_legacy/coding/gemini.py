"""Gemini coding compatibility shims."""

from __future__ import annotations

from hud.tools.coding import BashSession, BashTool, EditTool


class GeminiShellTool(BashTool):
    """Compatibility shim for old Gemini shell environment registrations."""

    def __init__(self, session: BashSession | None = None, cwd: str | None = None) -> None:
        super().__init__(
            session=session or (BashSession(cwd=cwd) if cwd is not None else None),
            name="bash",
            title="Bash Shell",
            description="Execute shell commands in a persistent bash session",
        )


class GeminiEditTool(EditTool):
    """Compatibility shim for old Gemini edit environment registrations."""

    def __init__(self, base_path: str = ".") -> None:
        super().__init__(
            base_path=base_path,
            name="edit",
            title="File Editor",
            description="View, create, and edit files with undo support",
        )


class GeminiWriteTool(EditTool):
    """Compatibility shim for old Gemini write_file environment registrations."""

    def __init__(self, base_path: str = ".") -> None:
        super().__init__(
            base_path=base_path,
            name="edit",
            title="File Editor",
            description="View, create, and edit files with undo support",
        )


__all__ = ["GeminiEditTool", "GeminiShellTool", "GeminiWriteTool"]
