"""Filesystem exploration tools for coding agents.

These tools provide read-only access to the filesystem, commonly used
by coding agents like OpenCode, Claude, Gemini, and others.

Two styles are available:
- OpenCode-style (default): ReadTool, GrepTool, GlobTool, ListTool
- Gemini CLI-style: GeminiReadTool, GeminiSearchTool, GeminiGlobTool, GeminiListTool

OpenCode-style usage:
    from hud.tools.filesystem import ReadTool, GrepTool, GlobTool, ListTool

    env = hud.Environment("my-agent")
    env.add_tool(ReadTool(base_path="./workspace"))
    env.add_tool(GrepTool(base_path="./workspace"))
    env.add_tool(GlobTool(base_path="./workspace"))
    env.add_tool(ListTool(base_path="./workspace"))

Gemini CLI-style usage:
    from hud.tools.filesystem import (
        GeminiReadTool, GeminiSearchTool, GeminiGlobTool, GeminiListTool
    )

    env = hud.Environment("my-agent")
    env.add_tool(GeminiReadTool(base_path="./workspace"))
    env.add_tool(GeminiSearchTool(base_path="./workspace"))
    env.add_tool(GeminiGlobTool(base_path="./workspace"))
    env.add_tool(GeminiListTool(base_path="./workspace"))
"""

# Gemini CLI-style tools
from hud.tools.filesystem.gemini import (
    GeminiGlobTool,
    GeminiListTool,
    GeminiReadTool,
    GeminiSearchTool,
)

# OpenCode-style tools (default)
from hud.tools.filesystem.glob import GlobTool
from hud.tools.filesystem.grep import GrepTool
from hud.tools.filesystem.list import ListTool
from hud.tools.filesystem.read import ReadTool

__all__ = [
    "GeminiGlobTool",
    "GeminiListTool",
    "GeminiReadTool",
    "GeminiSearchTool",
    "GlobTool",
    "GrepTool",
    "ListTool",
    "ReadTool",
]
