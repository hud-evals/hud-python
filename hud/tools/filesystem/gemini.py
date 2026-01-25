"""Gemini CLI-style filesystem tools.

These tools match the interface and output format of Gemini CLI:
https://github.com/google-gemini/gemini-cli

Key differences from OpenCode-style tools:
- read_file: Uses offset/limit (0-based), different truncation message
- search_file_content: Named differently, grouped output by file
- glob: Adds case_sensitive, respect_git_ignore options, absolute paths
- list_directory: Uses dir_path, ignore[] params, DIR/file output format
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

from mcp.types import ImageContent, TextContent  # noqa: TC002

from hud.tools.filesystem.base import (
    BaseGlobTool,
    BaseListTool,
    BaseReadTool,
    BaseSearchTool,
    FileMatch,
    ReadResult,
)
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType


class GeminiReadTool(BaseReadTool):
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
            base_path=base_path,
            max_lines=max_lines,
            name="read_file",
            title="ReadFile",
            description=(
                "Reads and returns the content of a specified file. If the file is large, "
                "the content will be truncated. Use 'offset' and 'limit' parameters to "
                "paginate through large files."
            ),
        )

    def format_output(self, result: ReadResult, path: str) -> str:
        """Format output in Gemini CLI style (truncation warning at top).

        Args:
            result: ReadResult from read_with_pagination
            path: Original path string for display

        Returns:
            Formatted output with truncation message if needed
        """
        file_content = "\n".join(result.lines)

        lines_shown_start = result.start_offset + 1
        lines_shown_end = result.start_offset + len(result.lines)
        has_more = result.total_lines > lines_shown_end or result.truncated

        is_partial = (result.start_offset > 0) or has_more or result.truncated_by_bytes

        if is_partial:
            next_offset = lines_shown_end
            return (
                f"IMPORTANT: The file content has been truncated.\n"
                f"Status: Showing lines {lines_shown_start}-{lines_shown_end} "
                f"of {result.total_lines} total lines.\n"
                f"Action: To read more, use 'offset' and 'limit' parameters. "
                f"Example: offset: {next_offset}.\n\n"
                f"--- FILE CONTENT (truncated) ---\n{file_content}"
            )
        else:
            return file_content

    async def __call__(
        self,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[TextContent | ImageContent]:
        """Read file contents with optional pagination.

        Args:
            file_path: Path to the file to read
            offset: 0-based line number to start reading from
            limit: Maximum number of lines to read

        Returns:
            List of TextContent (or ImageContent for images) with file contents
        """
        if not file_path or file_path.strip() == "":
            raise ToolError("The 'file_path' parameter must be non-empty.")

        path = self.resolve_path(file_path)

        if not path.exists():
            raise ToolError(f"File not found: {file_path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory, not a file: {file_path}")

        if offset is not None and offset < 0:
            raise ToolError("Offset must be a non-negative number")
        if limit is not None and limit <= 0:
            raise ToolError("Limit must be a positive number")

        # Handle images
        if self.is_image(path):
            result = self.read_image(path)
            return result.to_content_blocks()  # type: ignore[return-value]

        result = self.read_with_pagination(path, offset=offset or 0, limit=limit)
        output = self.format_output(result, file_path)

        return list(ContentResult(output=output).to_text_blocks())


class GeminiSearchTool(BaseSearchTool):
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
            base_path=base_path,
            max_results=max_results,
            max_files=max_files,
            name="search_file_content",
            title="Search",
            description=(
                "Search file contents using a regex pattern. "
                "Returns matching lines grouped by file with line numbers."
            ),
        )

    def format_output(self, matches: list[FileMatch], pattern: str) -> str:
        """Format output in Gemini CLI style (grouped by file).

        Args:
            matches: List of FileMatch objects
            pattern: Original search pattern

        Returns:
            Formatted output grouped by file
        """
        if not matches:
            return f"No matches found for pattern: {pattern}"

        truncated = len(matches) >= self._max_results

        # Group by file
        file_matches: dict[str, list[FileMatch]] = {}
        for match in matches:
            if match.path not in file_matches:
                file_matches[match.path] = []
            file_matches[match.path].append(match)

        lines = [f"Found {len(matches)} matches in {len(file_matches)} files"]
        lines.append("")

        for file_path, file_group in file_matches.items():
            lines.append(f"{file_path}:")
            lines.extend(f"  Line {match.line_num}: {match.line_text}" for match in file_group)
            lines.append("")

        if truncated:
            lines.append("(Results are truncated. Consider using a more specific pattern.)")

        return "\n".join(lines)

    async def __call__(
        self,
        pattern: str,
        dir_path: str | None = None,
        include: str | None = None,
    ) -> list[TextContent]:
        """Search file contents for a pattern.

        Args:
            pattern: Regex pattern to search for
            dir_path: Directory to search in (defaults to base path)
            include: Glob pattern to filter files (e.g., "*.py")

        Returns:
            List of TextContent with matching lines grouped by file
        """
        regex = self.compile_pattern(pattern)
        search_path = self.resolve_path(dir_path or ".")

        if not search_path.exists():
            raise ToolError(f"Directory not found: {dir_path or '.'}")

        matches = self.search_files(search_path, regex, include)
        output = self.format_output(matches, pattern)

        return ContentResult(output=output).to_text_blocks()


class GeminiGlobTool(BaseGlobTool):
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
            base_path=base_path,
            max_results=max_results,
            name="glob",
            title="Glob",
            description=(
                "Find files matching a glob pattern. Returns absolute file paths "
                "sorted alphabetically. Use ** for recursive matching."
            ),
        )

    def format_output(self, matches: list[tuple[Path, float]], pattern: str) -> str:
        """Format output in Gemini CLI style (absolute paths, alphabetical).

        Args:
            matches: List of (path, mtime) tuples
            pattern: Original glob pattern

        Returns:
            Formatted output with absolute paths
        """
        if not matches:
            return f"No files found matching: {pattern}"

        truncated = len(matches) >= self._max_results

        # Sort alphabetically (Gemini CLI behavior, not by mtime)
        sorted_matches = sorted(matches, key=lambda x: str(x[0]))

        # Return absolute paths (Gemini CLI format)
        abs_paths = [str(m.resolve()) for m, _mtime in sorted_matches]
        output = "\n".join(abs_paths)

        if truncated:
            output += "\n\n(Results are truncated. Consider using a more specific pattern.)"

        return output

    async def __call__(
        self,
        pattern: str,
        dir_path: str | None = None,
        case_sensitive: bool = True,
        respect_git_ignore: bool = True,
    ) -> list[TextContent]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern to match
            dir_path: Directory to search in (defaults to base path)
            case_sensitive: Whether matching is case-sensitive
            respect_git_ignore: Whether to respect .gitignore

        Returns:
            List of TextContent with matching file paths
        """
        base = self.resolve_path(dir_path or ".")

        if not base.exists():
            raise ToolError(f"Directory not found: {dir_path or '.'}")
        if not base.is_dir():
            raise ToolError(f"Not a directory: {dir_path or '.'}")

        matches = self.find_files(base, pattern, include_ignored=not respect_git_ignore)
        output = self.format_output(matches, pattern)

        return ContentResult(output=output).to_text_blocks()


class GeminiListTool(BaseListTool):
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
            base_path=base_path,
            max_entries=max_entries,
            name="list_directory",
            title="ListDirectory",
            description=(
                "List the contents of a directory. Returns files and subdirectories "
                "with DIR prefix for directories. Hidden files are excluded by default."
            ),
        )

    def format_output(
        self,
        entries: list[tuple[str, bool]],
        directory: Path,
        path_str: str,
    ) -> str:
        """Format output in Gemini CLI style (DIR/filename format).

        Args:
            entries: List of (relative_path, is_dir) tuples
            directory: Directory that was listed
            path_str: Original path string for display

        Returns:
            Formatted output with DIR prefix for directories
        """
        if not entries:
            return f"Empty directory: {path_str}"

        truncated = len(entries) >= self._max_entries

        # Format as DIR/filename (Gemini CLI format)
        lines = []
        for name, is_dir in entries:
            # Extract just the name (not full relative path)
            simple_name = name.rstrip("/").split("/")[-1]
            if is_dir:
                lines.append(f"DIR  {simple_name}")
            else:
                lines.append(f"     {simple_name}")

        output = "\n".join(lines)

        if truncated:
            output += f"\n\n(Limited to {self._max_entries} entries)"

        return output

    async def __call__(
        self,
        dir_path: str,
        ignore: list[str] | None = None,
    ) -> list[TextContent]:
        """List directory contents.

        Args:
            dir_path: Directory to list
            ignore: List of glob patterns to ignore

        Returns:
            List of TextContent with directory listing
        """
        if not dir_path:
            raise ToolError("The 'dir_path' parameter must be non-empty.")

        path = self.resolve_path(dir_path)

        if not path.exists():
            raise ToolError(f"Directory not found: {dir_path}")
        if not path.is_dir():
            raise ToolError(f"Path is not a directory: {dir_path}")

        entries = self.list_directory(path, ignore=ignore, recursive=False)
        output = self.format_output(entries, path, dir_path)

        return ContentResult(output=output).to_text_blocks()


__all__ = [
    "GeminiGlobTool",
    "GeminiListTool",
    "GeminiReadTool",
    "GeminiSearchTool",
]
