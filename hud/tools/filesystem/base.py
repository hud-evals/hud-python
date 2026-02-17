"""Base classes for filesystem tools.

Provides shared functionality for file reading, searching, and listing tools.
Two styles are supported:
- OpenCode-style: ReadTool, GrepTool, GlobTool, ListTool
- Gemini CLI-style: GeminiReadTool, GeminiSearchTool, GeminiGlobTool, GeminiListTool

Both styles share common operations but differ in:
- Parameter naming conventions
- Output formatting
- Truncation/pagination messages
"""

from __future__ import annotations

import base64
import fnmatch
import logging
import os
import re
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.types import ContentResult, ToolError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mcp.types import ContentBlock

    from hud.tools.native_types import NativeToolSpecs

LOGGER = logging.getLogger(__name__)

# Common constants
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_LINE_LENGTH = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
DEFAULT_MAX_RESULTS = 100
DEFAULT_MAX_FILES = 1000
DEFAULT_MAX_ENTRIES = 500

# Common ignore patterns
IGNORE_DIRS = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".git",
        "venv",
        ".venv",
        "dist",
        "build",
        "target",
        "vendor",
    }
)

# Image extensions
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"})

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
}


@dataclass
class FileMatch:
    """A single file match from search operations."""

    path: str
    line_num: int
    line_text: str
    mtime: float = 0.0


@dataclass
class ReadResult:
    """Result from reading a file."""

    lines: list[str]
    total_lines: int
    start_offset: int
    truncated: bool
    truncated_by_bytes: bool


class BaseFilesystemTool(BaseTool):
    """Abstract base for all filesystem tools.

    Provides common functionality:
    - Path resolution with security checks
    - File reading with encoding handling
    - Directory iteration with ignore patterns
    """

    native_specs: ClassVar[NativeToolSpecs] = {}

    _base_path: Path

    def __init__(
        self,
        base_path: str = ".",
        name: str = "filesystem",
        title: str = "Filesystem",
        description: str = "Filesystem tool",
    ) -> None:
        """Initialize filesystem tool.

        Args:
            base_path: Base directory for relative paths
            name: Tool name
            title: Tool title
            description: Tool description
        """
        super().__init__(env=None, name=name, title=title, description=description)
        self._base_path = Path(base_path).resolve()

    def resolve_path(self, path: str) -> Path:
        """Resolve and validate a path.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Resolved Path object
        """
        return resolve_path_safely(path, self._base_path)

    def is_ignored_dir(self, path: Path) -> bool:
        """Check if path contains an ignored directory."""
        return any(part in IGNORE_DIRS for part in path.parts)

    def is_hidden(self, path: Path) -> bool:
        """Check if path contains hidden files/directories."""
        return any(part.startswith(".") for part in path.parts)

    def read_file_content(self, path: Path) -> str:
        """Read file content with error handling.

        Args:
            path: Path to file

        Returns:
            File content as string

        Raises:
            ToolError: If file cannot be read
        """
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"Cannot read binary file: {path}") from None
        except PermissionError:
            raise ToolError(f"Permission denied: {path}") from None
        except FileNotFoundError:
            raise ToolError(f"File not found: {path}") from None

    def read_image(self, path: Path) -> ContentResult:
        """Read an image file and return base64 encoded content.

        Args:
            path: Path to image file

        Returns:
            ContentResult with image data
        """
        try:
            image_data = path.read_bytes()
            b64_content = base64.b64encode(image_data).decode("utf-8")
            mime = MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")
            return ContentResult(
                output=f"Image read successfully: {path}",
                system=f"data:{mime};base64,{b64_content[:100]}...",
            )
        except Exception as e:
            raise ToolError(f"Failed to read image: {e}") from None

    def is_image(self, path: Path) -> bool:
        """Check if path is an image file."""
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def iter_files(
        self,
        directory: Path,
        pattern: str | None = None,
        include_hidden: bool = False,
        include_ignored: bool = False,
        max_files: int = DEFAULT_MAX_FILES,
    ) -> Iterator[Path]:
        """Iterate over files in a directory.

        Args:
            directory: Directory to iterate
            pattern: Optional glob pattern to filter files
            include_hidden: Whether to include hidden files
            include_ignored: Whether to include ignored directories
            max_files: Maximum files to yield

        Yields:
            Path objects for matching files
        """
        count = 0
        iterator = directory.glob(pattern) if pattern else directory.rglob("*")

        for path in iterator:
            if count >= max_files:
                break

            if not path.is_file():
                continue

            if not include_hidden and self.is_hidden(path):
                continue

            if not include_ignored and self.is_ignored_dir(path):
                continue

            yield path
            count += 1

    def truncate_line(self, line: str, max_length: int = DEFAULT_MAX_LINE_LENGTH) -> str:
        """Truncate a line if it exceeds max length.

        Args:
            line: Line to truncate
            max_length: Maximum line length

        Returns:
            Truncated line with ellipsis if needed
        """
        if len(line) > max_length:
            return line[:max_length] + "..."
        return line

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> list[ContentBlock]:
        """Execute the filesystem operation."""
        ...


class BaseReadTool(BaseFilesystemTool):
    """Base class for file reading tools.

    Provides common file reading logic with pagination.
    Subclasses override format_output() to customize output style.
    """

    _max_lines: int
    _max_line_length: int
    _max_bytes: int

    def __init__(
        self,
        base_path: str = ".",
        max_lines: int = DEFAULT_MAX_LINES,
        max_line_length: int = DEFAULT_MAX_LINE_LENGTH,
        max_bytes: int = DEFAULT_MAX_BYTES,
        name: str = "read",
        title: str = "Read",
        description: str = "Read file contents",
    ) -> None:
        """Initialize read tool.

        Args:
            base_path: Base directory for relative paths
            max_lines: Maximum lines before truncation
            max_line_length: Maximum characters per line
            max_bytes: Maximum bytes to read
            name: Tool name
            title: Tool title
            description: Tool description
        """
        super().__init__(base_path=base_path, name=name, title=title, description=description)
        self._max_lines = max_lines
        self._max_line_length = max_line_length
        self._max_bytes = max_bytes

    def read_with_pagination(
        self,
        path: Path,
        offset: int = 0,
        limit: int | None = None,
    ) -> ReadResult:
        """Read file with pagination support.

        Args:
            path: Path to file
            offset: 0-based line offset
            limit: Maximum lines to read

        Returns:
            ReadResult with lines and metadata
        """
        content = self.read_file_content(path)
        lines = content.split("\n")
        total_lines = len(lines)

        read_limit = limit if limit is not None else self._max_lines
        start_offset = offset

        # Collect lines with byte limit
        result_lines: list[str] = []
        total_bytes = 0
        truncated_by_bytes = False

        for i in range(start_offset, min(total_lines, start_offset + read_limit)):
            line = lines[i]
            line = self.truncate_line(line, self._max_line_length)

            line_bytes = len(line.encode("utf-8")) + (1 if result_lines else 0)
            if total_bytes + line_bytes > self._max_bytes:
                truncated_by_bytes = True
                break

            result_lines.append(line)
            total_bytes += line_bytes

        # Check if truncated by line limit
        truncated = len(result_lines) >= self._max_lines

        return ReadResult(
            lines=result_lines,
            total_lines=total_lines,
            start_offset=start_offset,
            truncated=truncated,
            truncated_by_bytes=truncated_by_bytes,
        )

    @abstractmethod
    def format_output(self, result: ReadResult, path: str) -> str:
        """Format the read result as output string.

        Args:
            result: ReadResult from read_with_pagination
            path: Original path string for display

        Returns:
            Formatted output string
        """
        ...


class BaseSearchTool(BaseFilesystemTool):
    """Base class for file content search tools.

    Provides common regex search logic.
    Subclasses override format_output() to customize output style.
    """

    _max_results: int
    _max_files: int

    def __init__(
        self,
        base_path: str = ".",
        max_results: int = DEFAULT_MAX_RESULTS,
        max_files: int = DEFAULT_MAX_FILES,
        name: str = "grep",
        title: str = "Grep",
        description: str = "Search file contents",
    ) -> None:
        """Initialize search tool.

        Args:
            base_path: Base directory for relative paths
            max_results: Maximum matching lines
            max_files: Maximum files to search
            name: Tool name
            title: Tool title
            description: Tool description
        """
        super().__init__(base_path=base_path, name=name, title=title, description=description)
        self._max_results = max_results
        self._max_files = max_files

    def compile_pattern(self, pattern: str) -> re.Pattern[str]:
        """Compile a regex pattern.

        Args:
            pattern: Regex pattern string

        Returns:
            Compiled regex pattern

        Raises:
            ToolError: If pattern is invalid
        """
        if not pattern:
            raise ToolError("pattern is required")

        try:
            return re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Invalid regex pattern: {e}") from None

    def search_files(
        self,
        directory: Path,
        regex: re.Pattern[str],
        include: str | None = None,
    ) -> list[FileMatch]:
        """Search files for a pattern.

        Args:
            directory: Directory to search
            regex: Compiled regex pattern
            include: Optional glob pattern to filter files

        Returns:
            List of FileMatch objects
        """
        matches: list[FileMatch] = []

        # Collect files
        files: list[Path] = []
        if directory.is_file():
            files = [directory]
        else:
            for f in self.iter_files(directory, max_files=self._max_files):
                if include and not fnmatch.fnmatch(f.name, include):
                    continue
                files.append(f)

        # Search files
        for file in files:
            try:
                content = file.read_text(encoding="utf-8")
                mtime = os.path.getmtime(file)
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            try:
                rel_path = str(file.relative_to(self._base_path))
            except ValueError:
                rel_path = str(file)

            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    line_text = self.truncate_line(line.strip())
                    matches.append(
                        FileMatch(
                            path=rel_path,
                            line_num=i,
                            line_text=line_text,
                            mtime=mtime,
                        )
                    )

                    if len(matches) >= self._max_results:
                        return matches

        return matches

    @abstractmethod
    def format_output(self, matches: list[FileMatch], pattern: str) -> str:
        """Format search results as output string.

        Args:
            matches: List of FileMatch objects
            pattern: Original search pattern

        Returns:
            Formatted output string
        """
        ...


class BaseGlobTool(BaseFilesystemTool):
    """Base class for file globbing tools.

    Provides common glob logic.
    Subclasses override format_output() to customize output style.
    """

    _max_results: int

    def __init__(
        self,
        base_path: str = ".",
        max_results: int = DEFAULT_MAX_RESULTS,
        name: str = "glob",
        title: str = "Glob",
        description: str = "Find files by pattern",
    ) -> None:
        """Initialize glob tool.

        Args:
            base_path: Base directory for relative paths
            max_results: Maximum files to return
            name: Tool name
            title: Tool title
            description: Tool description
        """
        super().__init__(base_path=base_path, name=name, title=title, description=description)
        self._max_results = max_results

    def find_files(
        self,
        directory: Path,
        pattern: str,
        include_ignored: bool = False,
    ) -> list[tuple[Path, float]]:
        """Find files matching a glob pattern.

        Args:
            directory: Directory to search
            pattern: Glob pattern
            include_ignored: Whether to include ignored directories

        Returns:
            List of (path, mtime) tuples
        """
        if not pattern:
            raise ToolError("pattern is required")

        matches: list[tuple[Path, float]] = []

        try:
            for match in directory.glob(pattern):
                if self.is_hidden(match):
                    continue

                if not include_ignored and self.is_ignored_dir(match):
                    continue

                if not match.is_file():
                    continue

                try:
                    mtime = os.path.getmtime(match)
                except OSError:
                    mtime = 0

                matches.append((match, mtime))

                if len(matches) >= self._max_results:
                    break
        except Exception as e:
            raise ToolError(f"Invalid glob pattern: {e}") from None

        return matches

    @abstractmethod
    def format_output(self, matches: list[tuple[Path, float]], pattern: str) -> str:
        """Format glob results as output string.

        Args:
            matches: List of (path, mtime) tuples
            pattern: Original glob pattern

        Returns:
            Formatted output string
        """
        ...


class BaseListTool(BaseFilesystemTool):
    """Base class for directory listing tools.

    Provides common directory listing logic.
    Subclasses override format_output() to customize output style.
    """

    _max_entries: int

    def __init__(
        self,
        base_path: str = ".",
        max_entries: int = DEFAULT_MAX_ENTRIES,
        name: str = "list",
        title: str = "List",
        description: str = "List directory contents",
    ) -> None:
        """Initialize list tool.

        Args:
            base_path: Base directory for relative paths
            max_entries: Maximum entries to return
            name: Tool name
            title: Tool title
            description: Tool description
        """
        super().__init__(base_path=base_path, name=name, title=title, description=description)
        self._max_entries = max_entries

    def list_directory(
        self,
        directory: Path,
        ignore: list[str] | None = None,
        recursive: bool = True,
    ) -> list[tuple[str, bool]]:
        """List directory contents.

        Args:
            directory: Directory to list
            ignore: Patterns to ignore
            recursive: Whether to recurse into subdirectories

        Returns:
            List of (relative_path, is_dir) tuples
        """
        ignore_patterns = ignore or []
        entries: list[tuple[str, bool]] = []

        def should_ignore(name: str, is_dir: bool) -> bool:
            if name.startswith("."):
                return True
            for pattern in ignore_patterns:
                if pattern.endswith("/"):
                    if is_dir and fnmatch.fnmatch(name, pattern.rstrip("/")):
                        return True
                else:
                    if fnmatch.fnmatch(name, pattern):
                        return True
            return False

        def collect(dir_path: Path, prefix: str = "") -> None:
            if len(entries) >= self._max_entries:
                return

            try:
                items = list(dir_path.iterdir())
            except PermissionError:
                return

            # Sort: directories first, then files
            dirs = []
            files = []
            for item in items:
                if should_ignore(item.name, item.is_dir()):
                    continue
                if item.is_dir():
                    dirs.append(item)
                else:
                    files.append(item)

            dirs.sort(key=lambda x: x.name.lower())
            files.sort(key=lambda x: x.name.lower())

            for d in dirs:
                if len(entries) >= self._max_entries:
                    break
                rel = prefix + d.name + "/"
                entries.append((rel, True))
                if recursive:
                    collect(d, rel)

            for f in files:
                if len(entries) >= self._max_entries:
                    break
                entries.append((prefix + f.name, False))

        collect(directory)
        return entries

    @abstractmethod
    def format_output(
        self,
        entries: list[tuple[str, bool]],
        directory: Path,
        path_str: str,
    ) -> str:
        """Format directory listing as output string.

        Args:
            entries: List of (relative_path, is_dir) tuples
            directory: Directory that was listed
            path_str: Original path string for display

        Returns:
            Formatted output string
        """
        ...


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_FILES",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_LINE_LENGTH",
    "DEFAULT_MAX_RESULTS",
    "IGNORE_DIRS",
    "IMAGE_EXTENSIONS",
    "MIME_TYPES",
    "BaseFilesystemTool",
    "BaseGlobTool",
    "BaseListTool",
    "BaseReadTool",
    "BaseSearchTool",
    "FileMatch",
    "ReadResult",
]
