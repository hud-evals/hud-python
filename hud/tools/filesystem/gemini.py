"""Gemini CLI-style filesystem tools.

These tools match the interface and output format of Gemini CLI:
https://github.com/google-gemini/gemini-cli

Key differences from OpenCode-style tools:
- read_file: Uses offset/limit (0-based) instead of start_line/end_line (1-based)
- search_file_content: Named differently, grouped output by file
- glob: Adds case_sensitive, respect_git_ignore options
- list_directory: Uses dir_path, ignore[] params, DIR/file output format
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import ClassVar

from mcp.types import ContentBlock  # noqa: TC002

from hud.tools.base import BaseTool
from hud.tools.coding.utils import resolve_path_safely
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType


class GeminiReadTool(BaseTool):
    """Gemini CLI-style file reading tool.

    Reads file contents with offset/limit pagination (0-based).
    Matches Gemini CLI's read_file tool interface.

    Parameters:
        file_path: Path to the file to read (required)
        offset: 0-based line number to start reading from (optional)
        limit: Maximum number of lines to read (optional)

    Output includes truncation warnings with pagination hints.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(role="reader"),
    }

    _base_path: Path
    _max_lines: int

    def __init__(
        self,
        base_path: str = ".",
        max_lines: int = 2000,
    ) -> None:
        """Initialize GeminiReadTool.

        Args:
            base_path: Base directory for relative paths
            max_lines: Maximum lines before truncation (default: 2000)
        """
        super().__init__(
            env=None,
            name="read_file",
            title="ReadFile",
            description=(
                "Reads and returns the content of a specified file. If the file is large, "
                "the content will be truncated. Use 'offset' and 'limit' parameters to "
                "paginate through large files."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_lines = max_lines

    async def __call__(
        self,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ContentBlock]:
        """Read file contents with optional pagination.

        Args:
            file_path: Path to the file to read
            offset: 0-based line number to start reading from
            limit: Maximum number of lines to read

        Returns:
            List of ContentBlocks with file contents
        """
        if not file_path or file_path.strip() == "":
            raise ToolError("The 'file_path' parameter must be non-empty.")

        path = resolve_path_safely(file_path, self._base_path)

        if not path.exists():
            raise ToolError(f"File not found: {file_path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory, not a file: {file_path}")

        if offset is not None and offset < 0:
            raise ToolError("Offset must be a non-negative number")
        if limit is not None and limit <= 0:
            raise ToolError("Limit must be a positive number")

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"Cannot read binary file: {file_path}") from None
        except PermissionError:
            raise ToolError(f"Permission denied: {file_path}") from None

        lines = content.split("\n")
        total_lines = len(lines)

        # Apply offset/limit
        start_line = offset if offset is not None else 0
        end_line = min(start_line + limit, total_lines) if limit is not None else total_lines

        selected_lines = lines[start_line:end_line]
        lines_shown_start = start_line + 1
        lines_shown_end = start_line + len(selected_lines)

        # Check truncation
        truncated = False
        if len(selected_lines) > self._max_lines:
            selected_lines = selected_lines[: self._max_lines]
            lines_shown_end = start_line + self._max_lines
            truncated = True

        file_content = "\n".join(selected_lines)
        is_partial = (start_line > 0) or (end_line < total_lines) or truncated

        if is_partial:
            next_offset = lines_shown_end
            truncation_msg = f"""IMPORTANT: The file content has been truncated.
Status: Showing lines {lines_shown_start}-{lines_shown_end} of {total_lines} total lines.
Action: To read more, use 'offset' and 'limit' parameters. Example: offset: {next_offset}.

--- FILE CONTENT (truncated) ---
{file_content}"""
            return ContentResult(output=truncation_msg).to_content_blocks()
        else:
            return ContentResult(output=file_content).to_content_blocks()


class GeminiSearchTool(BaseTool):
    """Gemini CLI-style file content search tool.

    Searches file contents using regex patterns.
    Matches Gemini CLI's search_file_content tool interface.

    Parameters:
        pattern: Regex pattern to search for (required)
        dir_path: Directory to search in (optional, defaults to project root)
        include: Glob pattern to filter files (e.g., "*.py")
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(role="searcher"),
    }

    _base_path: Path
    _max_results: int
    _max_files: int

    def __init__(
        self,
        base_path: str = ".",
        max_results: int = 100,
        max_files: int = 1000,
    ) -> None:
        """Initialize GeminiSearchTool.

        Args:
            base_path: Base directory for relative paths
            max_results: Maximum matching lines to return
            max_files: Maximum files to search
        """
        super().__init__(
            env=None,
            name="search_file_content",
            title="Search",
            description=(
                "Search file contents using a regex pattern. "
                "Returns matching lines grouped by file with line numbers."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_results = max_results
        self._max_files = max_files

    async def __call__(
        self,
        pattern: str,
        dir_path: str | None = None,
        include: str | None = None,
    ) -> list[ContentBlock]:
        """Search file contents for a pattern.

        Args:
            pattern: Regex pattern to search for
            dir_path: Directory to search in (defaults to base path)
            include: Glob pattern to filter files (e.g., "*.py")

        Returns:
            List of ContentBlocks with matching lines grouped by file
        """
        if not pattern:
            raise ToolError("The 'pattern' parameter must be non-empty.")

        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Invalid regex pattern: {e}") from None

        search_path = resolve_path_safely(dir_path or ".", self._base_path)

        if not search_path.exists():
            raise ToolError(f"Directory not found: {dir_path or '.'}")

        # Collect files
        if search_path.is_file():
            files = [search_path]
        else:
            files = []
            for f in search_path.rglob("*"):
                if len(files) >= self._max_files:
                    break
                if not f.is_file():
                    continue
                if any(part.startswith(".") for part in f.parts):
                    continue
                if any(
                    part in ("node_modules", "__pycache__", ".git", "venv", ".venv")
                    for part in f.parts
                ):
                    continue
                if include and not fnmatch.fnmatch(f.name, include):
                    continue
                files.append(f)

        # Search and group by file
        file_matches: dict[str, list[tuple[int, str]]] = {}
        total_matches = 0

        for file in files:
            try:
                content = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            try:
                rel_path = str(file.relative_to(self._base_path))
            except ValueError:
                rel_path = str(file)

            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    if rel_path not in file_matches:
                        file_matches[rel_path] = []
                    file_matches[rel_path].append((i, line.strip()))
                    total_matches += 1

                    if total_matches >= self._max_results:
                        break

            if total_matches >= self._max_results:
                break

        # Format output (grouped by file like Gemini CLI)
        if not file_matches:
            output = f"No matches found for pattern: {pattern}"
        else:
            lines = [f"Found {total_matches} matches in {len(file_matches)} files"]
            lines.append("")

            for file_path, matches in file_matches.items():
                lines.append(f"{file_path}:")
                for line_num, line_text in matches:
                    lines.append(f"  Line {line_num}: {line_text}")
                lines.append("")

            if total_matches >= self._max_results:
                lines.append("(Results are truncated. Consider using a more specific pattern.)")

            output = "\n".join(lines)

        return ContentResult(output=output).to_content_blocks()


class GeminiGlobTool(BaseTool):
    """Gemini CLI-style file globbing tool.

    Finds files matching a glob pattern.
    Matches Gemini CLI's glob tool interface.

    Parameters:
        pattern: Glob pattern to match (required)
        dir_path: Directory to search in (optional)
        case_sensitive: Whether matching is case-sensitive (default: True)
        respect_git_ignore: Whether to respect .gitignore (default: True)
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(role="finder"),
    }

    _base_path: Path
    _max_results: int

    def __init__(
        self,
        base_path: str = ".",
        max_results: int = 100,
    ) -> None:
        """Initialize GeminiGlobTool.

        Args:
            base_path: Base directory for relative paths
            max_results: Maximum files to return (default: 100)
        """
        super().__init__(
            env=None,
            name="glob",
            title="Glob",
            description=(
                "Find files matching a glob pattern. Returns absolute file paths "
                "sorted alphabetically. Use ** for recursive matching."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_results = max_results

    async def __call__(
        self,
        pattern: str,
        dir_path: str | None = None,
        case_sensitive: bool = True,
        respect_git_ignore: bool = True,
    ) -> list[ContentBlock]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern to match
            dir_path: Directory to search in (defaults to base path)
            case_sensitive: Whether matching is case-sensitive
            respect_git_ignore: Whether to respect .gitignore

        Returns:
            List of ContentBlocks with matching file paths
        """
        if not pattern:
            raise ToolError("The 'pattern' parameter must be non-empty.")

        base = resolve_path_safely(dir_path or ".", self._base_path)

        if not base.exists():
            raise ToolError(f"Directory not found: {dir_path or '.'}")
        if not base.is_dir():
            raise ToolError(f"Not a directory: {dir_path or '.'}")

        # Find matches
        matches: list[Path] = []
        try:
            for match in base.glob(pattern):
                if any(part.startswith(".") for part in match.parts):
                    continue
                if respect_git_ignore and any(
                    part in ("node_modules", "__pycache__", "venv", ".venv", "dist", "build")
                    for part in match.parts
                ):
                    continue

                matches.append(match)
                if len(matches) >= self._max_results:
                    break
        except Exception as e:
            raise ToolError(f"Invalid glob pattern: {e}") from None

        # Sort alphabetically (Gemini CLI behavior)
        matches.sort()

        if not matches:
            output = f"No files found matching: {pattern}"
        else:
            # Return absolute paths (Gemini CLI format)
            abs_paths = [str(m.resolve()) for m in matches]
            output = "\n".join(abs_paths)

            if len(matches) >= self._max_results:
                output += "\n\n(Results are truncated. Consider using a more specific pattern.)"

        return ContentResult(output=output).to_content_blocks()


class GeminiListTool(BaseTool):
    """Gemini CLI-style directory listing tool.

    Lists directory contents with DIR/file format.
    Matches Gemini CLI's list_directory tool interface.

    Parameters:
        dir_path: Directory to list (required)
        ignore: List of glob patterns to ignore (optional)
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(role="lister"),
    }

    _base_path: Path
    _max_entries: int

    def __init__(
        self,
        base_path: str = ".",
        max_entries: int = 500,
    ) -> None:
        """Initialize GeminiListTool.

        Args:
            base_path: Base directory for relative paths
            max_entries: Maximum entries to return (default: 500)
        """
        super().__init__(
            env=None,
            name="list_directory",
            title="ListDirectory",
            description=(
                "List the contents of a directory. Returns files and subdirectories "
                "with DIR prefix for directories. Hidden files are excluded by default."
            ),
        )
        self._base_path = Path(base_path).resolve()
        self._max_entries = max_entries

    async def __call__(
        self,
        dir_path: str,
        ignore: list[str] | None = None,
    ) -> list[ContentBlock]:
        """List directory contents.

        Args:
            dir_path: Directory to list
            ignore: List of glob patterns to ignore

        Returns:
            List of ContentBlocks with directory listing
        """
        if not dir_path:
            raise ToolError("The 'dir_path' parameter must be non-empty.")

        path = resolve_path_safely(dir_path, self._base_path)

        if not path.exists():
            raise ToolError(f"Directory not found: {dir_path}")
        if not path.is_dir():
            raise ToolError(f"Path is not a directory: {dir_path}")

        ignore_patterns = ignore or []

        # Collect entries
        entries: list[tuple[str, bool]] = []  # (name, is_dir)

        for entry in path.iterdir():
            # Skip hidden
            if entry.name.startswith("."):
                continue

            # Check ignore patterns
            should_ignore = False
            for pattern in ignore_patterns:
                if fnmatch.fnmatch(entry.name, pattern):
                    should_ignore = True
                    break
            if should_ignore:
                continue

            entries.append((entry.name, entry.is_dir()))

            if len(entries) >= self._max_entries:
                break

        # Sort: directories first, then files, alphabetically
        entries.sort(key=lambda x: (not x[1], x[0].lower()))

        # Format output (Gemini CLI format: DIR/filename)
        if not entries:
            output = f"Empty directory: {dir_path}"
        else:
            lines = []
            for name, is_dir in entries:
                if is_dir:
                    lines.append(f"DIR  {name}")
                else:
                    lines.append(f"     {name}")

            output = "\n".join(lines)

            if len(entries) >= self._max_entries:
                output += f"\n\n(Limited to {self._max_entries} entries)"

        return ContentResult(output=output).to_content_blocks()


__all__ = [
    "GeminiGlobTool",
    "GeminiListTool",
    "GeminiReadTool",
    "GeminiSearchTool",
]
