"""Filesystem-based memory using grep/glob search.

This module implements the filesystem-as-context pattern:
- Store large outputs as files, keep only paths in context
- Use grep for search (more accurate than vector DB for code)
- No embedding model dependency
- Handles tasks far exceeding 128K tokens
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a grep/glob search."""

    path: str
    line_number: int | None = None
    content: str | None = None
    score: float = 1.0


@dataclass
class MemoryEntry:
    """An entry in the filesystem memory index."""

    key: str
    path: str
    summary: str
    content_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class FilesystemMemory:
    """Store large outputs as files, keep only paths in context.

    This implements the following principle: using the filesystem as unlimited
    external memory, with grep/glob for search instead of vector embeddings.

    Example:
        memory = FilesystemMemory(Path("./workspace"))

        # Store large content
        ref = await memory.store("research_results", long_content)
        # Returns: "[Stored at: ./workspace/research_results.txt]"

        # Search with grep
        results = await memory.search("authentication")
        # Returns files containing "authentication"

        # Read on demand
        content = await memory.read("./workspace/research_results.txt")
    """

    def __init__(self, workspace: Path) -> None:
        """Initialize filesystem memory.

        Args:
            workspace: Directory to store memory files
        """
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.index_file = self.workspace / ".memory_index.json"
        self._index: dict[str, MemoryEntry] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the memory index from disk."""
        if self.index_file.exists():
            try:
                data = json.loads(self.index_file.read_text())
                for key, entry_data in data.items():
                    entry_data["created_at"] = datetime.fromisoformat(entry_data["created_at"])
                    self._index[key] = MemoryEntry(**entry_data)
            except Exception as e:
                logger.warning(f"Failed to load memory index: {e}")
                self._index = {}

    def _save_index(self) -> None:
        """Save the memory index to disk."""
        data = {}
        for key, entry in self._index.items():
            entry_dict = {
                "key": entry.key,
                "path": entry.path,
                "summary": entry.summary,
                "content_hash": entry.content_hash,
                "created_at": entry.created_at.isoformat(),
                "size_bytes": entry.size_bytes,
                "metadata": entry.metadata,
            }
            data[key] = entry_dict

        self.index_file.write_text(json.dumps(data, indent=2))

    def _hash_content(self, content: str) -> str:
        """Generate a hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def store(
        self,
        key: str,
        content: str,
        extension: str = "txt",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store content as file, return path reference.

        Args:
            key: Unique key for this content
            content: The content to store
            extension: File extension (default: txt)
            metadata: Optional metadata to store with the entry

        Returns:
            Reference string to include in context
        """
        # Sanitize key for filename
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        path = self.workspace / f"{safe_key}.{extension}"

        # Write content
        path.write_text(content)

        # Create index entry
        content_hash = self._hash_content(content)
        summary = content[:200].replace("\n", " ").strip()
        if len(content) > 200:
            summary += "..."

        entry = MemoryEntry(
            key=key,
            path=str(path),
            summary=summary,
            content_hash=content_hash,
            size_bytes=len(content.encode()),
            metadata=metadata or {},
        )
        self._index[key] = entry
        self._save_index()

        logger.debug(f"Stored {len(content)} bytes at {path}")
        return f"[Stored at: {path}]"

    async def read(self, path: str) -> str:
        """Read content from a file path.

        Args:
            path: Path to the file

        Returns:
            File content
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text()

    async def get(self, key: str) -> str | None:
        """Get content by key.

        Args:
            key: The key used when storing

        Returns:
            Content if found, None otherwise
        """
        entry = self._index.get(key)
        if entry is None:
            return None
        return await self.read(entry.path)

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search files using grep.

        This is more accurate than vector search for code/text.

        Args:
            query: Search query (passed to grep)
            max_results: Maximum number of results

        Returns:
            List of matching files with context
        """
        try:
            # Use grep -r for recursive search, -l for file names only
            # -i for case insensitive
            proc = await asyncio.create_subprocess_exec(
                "grep",
                "-r",
                "-i",
                "-l",
                "--include=*.txt",
                "--include=*.py",
                "--include=*.json",
                "--include=*.md",
                query,
                str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode not in (0, 1):  # 1 means no matches
                logger.warning(f"grep error: {stderr.decode()}")
                return []

            results = []
            for line in stdout.decode().strip().split("\n"):
                if line:
                    results.append(SearchResult(path=line))
                    if len(results) >= max_results:
                        break

            return results

        except FileNotFoundError:
            logger.warning("grep not available, falling back to Python search")
            return await self._python_search(query, max_results)

    async def _python_search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Fallback search using Python (when grep not available)."""
        results = []
        query_lower = query.lower()

        for path in self.workspace.rglob("*"):
            if path.is_file() and path.suffix in (".txt", ".py", ".json", ".md"):
                try:
                    content = path.read_text()
                    if query_lower in content.lower():
                        results.append(SearchResult(path=str(path)))
                        if len(results) >= max_results:
                            break
                except Exception:
                    continue

        return results

    async def search_with_context(
        self,
        query: str,
        context_lines: int = 3,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """Search with surrounding context lines.

        Args:
            query: Search query
            context_lines: Number of lines before/after match
            max_results: Maximum number of results

        Returns:
            Results with line numbers and context
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "grep",
                "-r",
                "-i",
                "-n",
                f"-C{context_lines}",
                "--include=*.txt",
                "--include=*.py",
                "--include=*.json",
                "--include=*.md",
                query,
                str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode not in (0, 1):
                return []

            results = []
            current_file = None
            current_content = []

            for line in stdout.decode().strip().split("\n"):
                if not line:
                    continue

                # Parse grep output: file:line:content or file-line-content
                if ":" in line:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        file_path, line_num, content = parts[0], parts[1], parts[2]
                        if current_file != file_path:
                            if current_file and current_content:
                                results.append(
                                    SearchResult(
                                        path=current_file,
                                        content="\n".join(current_content),
                                    )
                                )
                            current_file = file_path
                            current_content = []
                        current_content.append(f"{line_num}: {content}")

                        if len(results) >= max_results:
                            break

            # Don't forget the last file
            if current_file and current_content:
                results.append(
                    SearchResult(
                        path=current_file,
                        content="\n".join(current_content),
                    )
                )

            return results[:max_results]

        except FileNotFoundError:
            # Fallback to simple search
            return await self.search(query, max_results)

    async def glob(self, pattern: str) -> list[str]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.txt")

        Returns:
            List of matching file paths
        """
        return [str(p) for p in self.workspace.glob(pattern)]

    async def list_files(self) -> list[str]:
        """List all files in the workspace."""
        return [str(p) for p in self.workspace.rglob("*") if p.is_file()]

    async def get_file_hashes(self) -> dict[str, str]:
        """Get hash of all files for checkpointing.

        Returns:
            Dict of path -> content hash
        """
        hashes = {}
        for path in self.workspace.rglob("*"):
            if path.is_file() and path.name != ".memory_index.json":
                try:
                    content = path.read_text()
                    hashes[str(path)] = self._hash_content(content)
                except Exception:
                    continue
        return hashes

    def get_index(self) -> dict[str, Any]:
        """Get the memory index for checkpointing."""
        return {
            key: {
                "key": entry.key,
                "path": entry.path,
                "summary": entry.summary,
                "content_hash": entry.content_hash,
                "created_at": entry.created_at.isoformat(),
                "size_bytes": entry.size_bytes,
                "metadata": entry.metadata,
            }
            for key, entry in self._index.items()
        }

    def restore_index(self, index_data: dict[str, Any]) -> None:
        """Restore the memory index from checkpoint."""
        self._index = {}
        for key, entry_data in index_data.items():
            entry_data["created_at"] = datetime.fromisoformat(entry_data["created_at"])
            self._index[key] = MemoryEntry(**entry_data)
        self._save_index()

    def get_summary(self, key: str) -> str | None:
        """Get just the summary for a key (for context)."""
        entry = self._index.get(key)
        return entry.summary if entry else None

    def __repr__(self) -> str:
        return f"FilesystemMemory(workspace={self.workspace}, entries={len(self._index)})"


__all__ = ["FilesystemMemory", "SearchResult", "MemoryEntry"]

