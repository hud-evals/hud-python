"""Filesystem exploration tools for coding agents.

These tools provide read-only access to the filesystem, commonly used
by coding agents like OpenCode, Claude, and others.

Usage:
    from hud.tools.filesystem import ReadTool, GrepTool, GlobTool, ListTool

    env = hud.Environment("my-agent")
    env.add_tool(ReadTool(base_path="./workspace"))
    env.add_tool(GrepTool(base_path="./workspace"))
    env.add_tool(GlobTool(base_path="./workspace"))
    env.add_tool(ListTool(base_path="./workspace"))
"""

from hud.tools.filesystem.glob import GlobTool
from hud.tools.filesystem.grep import GrepTool
from hud.tools.filesystem.list import ListTool
from hud.tools.filesystem.read import ReadTool

__all__ = [
    "GlobTool",
    "GrepTool",
    "ListTool",
    "ReadTool",
]
