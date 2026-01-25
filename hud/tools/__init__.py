"""HUD tools for computer control, file editing, and bash commands.

For coding tools (shell, bash, edit, apply_patch), import from:
    from hud.tools.coding import BashTool, ShellTool, EditTool, ApplyPatchTool

For filesystem tools (read, grep, glob, list), import from:
    from hud.tools.filesystem import ReadTool, GrepTool, GlobTool, ListTool

For computer tools, import from:
    from hud.tools.computer import AnthropicComputerTool, OpenAIComputerTool
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Base classes and types
from .agent import AgentTool
from .base import BaseHub, BaseTool
from .hosted import (
    CodeExecutionTool,
    GoogleSearchTool,
    HostedTool,
    UrlContextTool,
    WebFetchTool,
    WebSearchTool,
)
from .memory import MemoryTool
from .native_types import NativeToolSpec, NativeToolSpecs
from .playwright import PlaywrightTool
from .response import ResponseTool
from .submit import SubmitTool

if TYPE_CHECKING:
    from .coding import (
        ApplyPatchTool,
        BashTool,
        EditTool,
        GeminiEditTool,
        GeminiShellTool,
        ShellTool,
    )
    from .computer import (
        AnthropicComputerTool,
        GeminiComputerTool,
        HudComputerTool,
        OpenAIComputerTool,
        QwenComputerTool,
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
    "CodeExecutionTool",
    "EditTool",
    "GeminiComputerTool",
    "GeminiEditTool",
    "GeminiShellTool",
    "GlobTool",
    "GoogleSearchTool",
    "GrepTool",
    "HostedTool",
    "HudComputerTool",
    "ListTool",
    "MemoryTool",
    "NativeToolSpec",
    "NativeToolSpecs",
    "OpenAIComputerTool",
    "PlaywrightTool",
    "QwenComputerTool",
    "ReadTool",
    "ResponseTool",
    "ShellTool",
    "SubmitTool",
    "UrlContextTool",
    "WebFetchTool",
    "WebSearchTool",
]


def __getattr__(name: str) -> Any:
    """Lazy import tools to avoid heavy imports unless needed."""
    # Computer tools
    if name in (
        "AnthropicComputerTool",
        "HudComputerTool",
        "OpenAIComputerTool",
        "GeminiComputerTool",
        "QwenComputerTool",
    ):
        from . import computer

        return getattr(computer, name)

    # Coding tools
    if name in (
        "BashTool",
        "EditTool",
        "ShellTool",
        "ApplyPatchTool",
        "GeminiShellTool",
        "GeminiEditTool",
    ):
        from . import coding

        return getattr(coding, name)

    # Filesystem tools
    if name in (
        "ReadTool",
        "GrepTool",
        "GlobTool",
        "ListTool",
    ):
        from . import filesystem

        return getattr(filesystem, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
