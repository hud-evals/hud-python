"""Base classes for memory tools.

Memory tools provide persistent or session-based storage for agents.
Three paradigms are supported:

1. Session Memory (MemoryTool):
   - In-memory or vector DB backed
   - add/search interface
   - Lost on session end

2. File-based Memory (ClaudeMemoryTool, GeminiMemoryTool):
   - Persistent across sessions
   - File system storage
   - Different command sets per agent

3. Fact-based Memory (GeminiMemoryTool):
   - Appends facts to markdown file
   - Simple save_memory(fact) interface
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from hud.tools.base import BaseTool

if TYPE_CHECKING:
    from mcp.types import ContentBlock

    from hud.tools.native_types import NativeToolSpecs

LOGGER = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry with text and metadata."""

    text: str
    metadata: dict[str, Any]
    tokens: set[str]


class BaseMemoryTool(BaseTool):
    """Abstract base for all memory tools.

    Subclasses implement either:
    - Session-based memory (add/search)
    - File-based memory (view/create/edit/delete)
    - Fact-based memory (save)
    """

    native_specs: ClassVar[NativeToolSpecs] = {}

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> list[ContentBlock]:
        """Execute a memory operation."""
        ...


class BaseFileMemoryTool(BaseMemoryTool):
    """Base class for file-based memory tools.

    Provides common functionality for tools that store memories as files:
    - Path resolution with security checks
    - Directory management
    - File reading/writing utilities
    """

    _base_path: Path
    _memory_section_header: str

    def __init__(
        self,
        base_path: str | Path = ".",
        memory_section_header: str = "## Memories",
    ) -> None:
        """Initialize file-based memory tool.

        Args:
            base_path: Base directory for memory files
            memory_section_header: Markdown header for memory section
        """
        super().__init__(env=None, name="memory", title="Memory")
        self._base_path = Path(base_path).resolve()
        self._memory_section_header = memory_section_header

        # Ensure base directory exists
        self._base_path.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str) -> Path:
        """Resolve and validate a path within the memory directory.

        Prevents directory traversal attacks.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path escapes the base directory
        """
        relative = path.lstrip("/") if path.startswith("/") else path
        resolved = (self._base_path / relative).resolve()

        # Security check - prevent traversal
        try:
            resolved.relative_to(self._base_path)
        except ValueError:
            raise ValueError(f"Path traversal detected: {path}") from None

        return resolved

    def read_memory_file(self, path: Path) -> str:
        """Read memory file contents.

        Args:
            path: Path to file

        Returns:
            File contents as string, or empty string if file doesn't exist
        """
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except Exception as e:
            LOGGER.warning("Failed to read memory file %s: %s", path, e)
            return ""

    def write_memory_file(self, path: Path, content: str) -> None:
        """Write content to memory file.

        Creates parent directories if needed.

        Args:
            path: Path to file
            content: Content to write
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class BaseSessionMemoryTool(BaseMemoryTool):
    """Base class for session-based memory tools.

    Provides common functionality for in-memory or vector DB backed memory:
    - Add entries
    - Search/query entries
    - Token-based similarity (fallback)
    """

    _entries: list[MemoryEntry]

    def __init__(self) -> None:
        """Initialize session memory tool."""
        super().__init__(env=None, name="memory", title="Memory")
        self._entries = []

    @staticmethod
    def tokenize(text: str) -> set[str]:
        """Simple tokenization for similarity search."""
        return {t.lower() for t in text.split() if t}

    @staticmethod
    def jaccard_similarity(a: set[str], b: set[str]) -> float:
        """Calculate Jaccard similarity between token sets."""
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def add_entry(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a memory entry.

        Args:
            text: Text content to store
            metadata: Optional metadata dictionary
        """
        self._entries.append(
            MemoryEntry(
                text=text,
                metadata=metadata or {},
                tokens=self.tokenize(text),
            )
        )

    def search_entries(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search memory entries by similarity.

        Args:
            query: Search query
            top_k: Maximum results to return

        Returns:
            List of matching entries sorted by relevance
        """
        q_tokens = self.tokenize(query)
        scored = [
            (entry, self.jaccard_similarity(q_tokens, entry.tokens)) for entry in self._entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored[:top_k] if score > 0.0]


__all__ = [
    "BaseFileMemoryTool",
    "BaseMemoryTool",
    "BaseSessionMemoryTool",
    "MemoryEntry",
]
