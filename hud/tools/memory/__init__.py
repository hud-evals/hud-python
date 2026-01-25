"""Memory tools for persistent and session-based storage.

This module provides memory tools for different agent types:

Session Memory (short-term):
    - SessionMemoryTool: In-memory or Qdrant-backed add/search

File-based Memory (persistent):
    - ClaudeMemoryTool: File operations under /memories directory
    - GeminiMemoryTool: Simple fact storage in GEMINI.md

Usage:
    # Session memory (lost on restart)
    from hud.tools.memory import SessionMemoryTool
    tool = SessionMemoryTool()
    await tool(action="add", text="User prefers dark mode")
    await tool(action="search", text="user preferences")

    # Claude persistent memory
    from hud.tools.memory import ClaudeMemoryTool
    tool = ClaudeMemoryTool(memories_dir="/memories")
    await tool(command="create", path="/memories/notes.md", file_text="...")

    # Gemini persistent memory
    from hud.tools.memory import GeminiMemoryTool
    tool = GeminiMemoryTool(memory_dir="./workspace")
    await tool(fact="User prefers tabs over spaces")
"""

from hud.tools.memory.base import (
    BaseFileMemoryTool,
    BaseMemoryTool,
    BaseSessionMemoryTool,
    MemoryEntry,
)
from hud.tools.memory.claude import ClaudeMemoryCommand, ClaudeMemoryTool
from hud.tools.memory.gemini import GeminiMemoryTool
from hud.tools.memory.session import MemoryTool, SessionMemoryTool

__all__ = [
    # Base classes
    "BaseFileMemoryTool",
    "BaseMemoryTool",
    "BaseSessionMemoryTool",
    "MemoryEntry",
    # Claude
    "ClaudeMemoryCommand",
    "ClaudeMemoryTool",
    # Gemini
    "GeminiMemoryTool",
    # Session (default)
    "MemoryTool",
    "SessionMemoryTool",
]
