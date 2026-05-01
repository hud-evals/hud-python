"""Compatibility shims for old public tool names."""

from __future__ import annotations

import sys
from importlib import import_module

from hud.tools._legacy.coding import (
    ApplyPatchTool,
    DiffError,
    GeminiEditTool,
    GeminiShellTool,
    GeminiWriteTool,
    ShellTool,
)
from hud.tools._legacy.computer import (
    AnthropicComputerTool,
    GeminiComputerTool,
    GLMComputerTool,
    HudComputerTool,
    OpenAIComputerTool,
    QwenComputerTool,
)
from hud.tools._legacy.filesystem import (
    GeminiGlobTool,
    GeminiListTool,
    GeminiReadManyTool,
    GeminiReadTool,
    GeminiSearchTool,
    GlobTool,
    GrepTool,
    ListTool,
    ReadTool,
)
from hud.tools._legacy.memory import ClaudeMemoryCommand, ClaudeMemoryTool, GeminiMemoryTool

_DEEP_MODULE_ALIASES = {
    "hud.tools.coding.apply_patch": "hud.tools._legacy.coding.apply_patch",
    "hud.tools.coding.gemini_edit": "hud.tools._legacy.coding.gemini",
    "hud.tools.coding.gemini_shell": "hud.tools._legacy.coding.gemini",
    "hud.tools.coding.gemini_write": "hud.tools._legacy.coding.gemini",
    "hud.tools.coding.shell": "hud.tools._legacy.coding.shell",
    "hud.tools.computer.anthropic": "hud.tools._legacy.computer.anthropic",
    "hud.tools.computer.gemini": "hud.tools._legacy.computer.gemini",
    "hud.tools.computer.glm": "hud.tools._legacy.computer.glm",
    "hud.tools.computer.hud": "hud.tools._legacy.computer.hud",
    "hud.tools.computer.openai": "hud.tools._legacy.computer.openai",
    "hud.tools.computer.qwen": "hud.tools._legacy.computer.qwen",
    "hud.tools.filesystem.gemini": "hud.tools._legacy.filesystem.gemini",
    "hud.tools.filesystem.glob": "hud.tools._legacy.filesystem.glob",
    "hud.tools.filesystem.grep": "hud.tools._legacy.filesystem.grep",
    "hud.tools.filesystem.list": "hud.tools._legacy.filesystem.list",
    "hud.tools.filesystem.read": "hud.tools._legacy.filesystem.read",
}

_PARENT_SYMBOL_ALIASES = {
    "hud.tools.coding": (
        "ApplyPatchTool",
        "GeminiEditTool",
        "GeminiShellTool",
        "GeminiWriteTool",
        "ShellTool",
    ),
    "hud.tools.computer": (
        "AnthropicComputerTool",
        "GLMComputerTool",
        "GeminiComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
        "QwenComputerTool",
    ),
    "hud.tools.filesystem": (
        "GeminiGlobTool",
        "GeminiListTool",
        "GeminiReadManyTool",
        "GeminiReadTool",
        "GeminiSearchTool",
    ),
}


def install_legacy_aliases() -> None:
    """Install old import paths as aliases to this compatibility package tree."""
    for public_name, legacy_name in _DEEP_MODULE_ALIASES.items():
        module = import_module(legacy_name)
        sys.modules.setdefault(public_name, module)
        parent_name, _, child_name = public_name.rpartition(".")
        if parent_name:
            setattr(import_module(parent_name), child_name, module)

    for parent_name, symbols in _PARENT_SYMBOL_ALIASES.items():
        parent = import_module(parent_name)
        for symbol in symbols:
            setattr(parent, symbol, globals()[symbol])


__all__ = [
    "AnthropicComputerTool",
    "ApplyPatchTool",
    "ClaudeMemoryCommand",
    "ClaudeMemoryTool",
    "DiffError",
    "GLMComputerTool",
    "GeminiComputerTool",
    "GeminiEditTool",
    "GeminiGlobTool",
    "GeminiListTool",
    "GeminiMemoryTool",
    "GeminiReadManyTool",
    "GeminiReadTool",
    "GeminiSearchTool",
    "GeminiShellTool",
    "GeminiWriteTool",
    "GlobTool",
    "GrepTool",
    "HudComputerTool",
    "ListTool",
    "OpenAIComputerTool",
    "QwenComputerTool",
    "ReadTool",
    "ShellTool",
    "install_legacy_aliases",
]
