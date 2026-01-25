"""Coding tools for OpenAI-native shell execution and file patching.

This module provides OpenAI-native tools:

Shell Tools (command execution):
- ShellTool: OpenAI-native shell tool with auto-restart

File Patching Tools:
- ApplyPatchTool: OpenAI-native V4A diff patch tool

For Claude-native tools, use:
- hud.tools.BashTool (Claude bash)
- hud.tools.EditTool (Claude str_replace editor)
"""

from hud.tools.coding.apply_patch import ApplyPatchResult, ApplyPatchTool, DiffError
from hud.tools.coding.session import BashSession, ShellCallOutcome, ShellCommandOutput
from hud.tools.coding.shell import ShellResult, ShellTool

__all__ = [
    # OpenAI-native tools
    "ShellTool",
    "ApplyPatchTool",
    # Session management
    "BashSession",
    "ShellCallOutcome",
    "ShellCommandOutput",
    # Result types
    "ShellResult",
    "ApplyPatchResult",
    # Errors
    "DiffError",
]
