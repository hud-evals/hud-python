"""Legacy apply_patch import path."""

from __future__ import annotations

from hud.tools.coding import EditTool


class DiffError(ValueError):
    """Compatibility error type for old imports."""


class ApplyPatchTool(EditTool):
    """Backward-compatible import name for EditTool."""

    def __init__(self, base_path: str = ".") -> None:
        super().__init__(
            base_path=base_path,
            name="edit",
            title="File Editor",
            description="View, create, and edit files with undo support",
        )

__all__ = ["ApplyPatchTool", "DiffError"]
