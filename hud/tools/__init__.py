"""HUD tools for computer control, file editing, and bash commands.

For coding tools, import from:
    from hud.tools.coding import BashTool, EditTool

For filesystem tools, import from:
    from hud.tools.filesystem import ReadTool, GrepTool, GlobTool, ListTool

For legacy compatibility shims, import from:
    from hud.tools import ShellTool, ApplyPatchTool

For computer tools, import from:
    from hud.tools.computer import ComputerTool
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._legacy import install_legacy_aliases as _install_legacy_aliases

# Base classes and types
from .agent import AgentTool
from .base import BaseHub, BaseTool
from .memory import (
    MemoryTool,
)
from .playwright import PlaywrightTool
from .submit import SubmitTool

if TYPE_CHECKING:
    from ._legacy import (
        AnthropicComputerTool,
        ApplyPatchTool,
        ClaudeMemoryTool,
        GeminiComputerTool,
        GeminiGlobTool,
        GeminiListTool,
        GeminiMemoryTool,
        GeminiReadManyTool,
        GeminiReadTool,
        GeminiSearchTool,
        GLMComputerTool,
        HudComputerTool,
        OpenAIComputerTool,
        QwenComputerTool,
        ShellTool,
    )
    from .coding import (
        BashTool,
        EditTool,
    )
    from .computer import (
        ComputerTool,
    )
    from .filesystem import (
        GlobTool,
        GrepTool,
        ListTool,
        ReadTool,
    )

__all__ = [
    "AgentTool",
    "AnthropicComputerTool",
    "ApplyPatchTool",
    "BaseHub",
    "BaseTool",
    "BashTool",
    "ClaudeMemoryTool",
    "ComputerTool",
    "EditTool",
    "GLMComputerTool",
    "GeminiComputerTool",
    "GeminiGlobTool",
    "GeminiListTool",
    "GeminiMemoryTool",
    "GeminiReadManyTool",
    "GeminiReadTool",
    "GeminiSearchTool",
    "GlobTool",
    "GrepTool",
    "HudComputerTool",
    "ListTool",
    "MemoryTool",
    "OpenAIComputerTool",
    "PlaywrightTool",
    "QwenComputerTool",
    "ReadTool",
    "ShellTool",
    "SubmitTool",
]


def __getattr__(name: str) -> Any:
    """Lazy import tools to avoid heavy imports unless needed."""
    # Computer tools
    if name == "ComputerTool":
        from . import computer

        return getattr(computer, name)

    # Coding tools
    if name in ("BashTool", "EditTool"):
        from . import coding

        return getattr(coding, name)

    # Filesystem tools
    if name in ("ReadTool", "GrepTool", "GlobTool", "ListTool"):
        from . import filesystem

        return getattr(filesystem, name)

    # Compatibility shims
    if name in (
        "ApplyPatchTool",
        "ShellTool",
        "ClaudeMemoryTool",
        "AnthropicComputerTool",
        "GLMComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
        "GeminiComputerTool",
        "QwenComputerTool",
        "GeminiReadTool",
        "GeminiReadManyTool",
        "GeminiSearchTool",
        "GeminiGlobTool",
        "GeminiListTool",
        "GeminiMemoryTool",
    ):
        from . import _legacy

        return getattr(_legacy, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


_install_legacy_aliases()
del _install_legacy_aliases
