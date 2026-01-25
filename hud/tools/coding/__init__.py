"""Coding tools for shell execution and file editing.

All coding-related tools (shell, bash, edit, apply_patch) are centralized here.

Usage:
    from hud.tools.coding import BashTool, ShellTool, EditTool, ApplyPatchTool

Claude-native tools:
    - BashTool: Persistent bash shell with manual restart (bash_20250124)
    - EditTool: str_replace-based file editor (text_editor_20250728)

OpenAI-native tools:
    - ShellTool: Shell with auto-restart and dynamic timeout (shell)
    - ApplyPatchTool: V4A diff-based file patching (apply_patch)

Gemini/Generic tools (function calling only):
    - GeminiShellTool: Simple run_shell_command style
    - GeminiEditTool: Simple edit style with instruction
"""

from hud.tools.coding.apply_patch import ApplyPatchResult, ApplyPatchTool, DiffError
from hud.tools.coding.bash import BashTool, ClaudeBashSession, _BashSession
from hud.tools.coding.edit import Command, EditTool
from hud.tools.coding.gemini_edit import GeminiEditTool
from hud.tools.coding.gemini_shell import GeminiShellOutput, GeminiShellTool
from hud.tools.coding.session import BashSession, ShellCallOutcome, ShellCommandOutput
from hud.tools.coding.shell import ShellResult, ShellTool
from hud.tools.coding.utils import (
    SNIPPET_LINES,
    make_snippet,
    maybe_truncate,
    read_file_async,
    read_file_sync,
    validate_path,
    write_file_async,
    write_file_sync,
)

# Memory tools (re-exported from hud.tools.memory for backwards compatibility)
from hud.tools.memory import ClaudeMemoryCommand as MemoryCommand
from hud.tools.memory import ClaudeMemoryTool

__all__ = [
    "ApplyPatchResult",
    "ApplyPatchTool",
    "BashSession",
    "BashTool",
    "ClaudeBashSession",
    "ClaudeMemoryTool",
    "Command",
    "DiffError",
    "EditTool",
    "GeminiEditTool",
    "GeminiShellOutput",
    "GeminiShellTool",
    "MemoryCommand",
    "SNIPPET_LINES",
    "ShellCallOutcome",
    "ShellCommandOutput",
    "ShellResult",
    "ShellTool",
    "_BashSession",
    "make_snippet",
    "maybe_truncate",
    "read_file_async",
    "read_file_sync",
    "validate_path",
    "write_file_async",
    "write_file_sync",
]
