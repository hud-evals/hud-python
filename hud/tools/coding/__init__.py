"""Coding environment tools for shell execution and file editing."""

from __future__ import annotations

from typing import Any

from .bash import (
    BashTool,
    BashToolSession,
    ClaudeBashSession,
    _BashSession,
)
from .edit import Command, EditTool
from .session import BashSession, ShellCallOutcome, ShellCommandOutput
from .utils import (
    SNIPPET_LINES,
    make_snippet,
    maybe_truncate,
    read_file_async,
    read_file_sync,
    validate_path,
    write_file_async,
    write_file_sync,
)

__all__ = [
    "SNIPPET_LINES",
    "BashSession",
    "BashTool",
    "BashToolSession",
    "ClaudeBashSession",
    "Command",
    "EditTool",
    "ShellCallOutcome",
    "ShellCommandOutput",
    "_BashSession",
    "make_snippet",
    "maybe_truncate",
    "read_file_async",
    "read_file_sync",
    "validate_path",
    "write_file_async",
    "write_file_sync",
]


def __getattr__(name: str) -> Any:
    """v5 names removed in v6 (``ApplyPatchTool``, ``ShellTool``, …) resolve to no-ops."""
    from hud._legacy import resolve_legacy_name

    return resolve_legacy_name(__name__, name)
