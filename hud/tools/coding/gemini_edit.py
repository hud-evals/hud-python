"""Gemini-style edit tool implementation.

Based on Gemini CLI's replace tool:
https://github.com/google-gemini/gemini-cli
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

from mcp.types import ContentBlock  # noqa: TC002 - used at runtime by FunctionTool

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType

from .utils import (
    SNIPPET_LINES,
    read_file_sync,
    write_file_sync,
)


def _escape_regex(s: str) -> str:
    """Escape regex special characters."""
    return re.sub(r"[.*+?^${}()|[\]\\]", r"\\\g<0>", s)


def _flexible_match(content: str, old_string: str, new_string: str) -> tuple[str, int]:
    """Attempt flexible whitespace-insensitive matching.

    Matches Gemini CLI behavior: strips each line and compares,
    preserves original indentation in replacement.
    """
    source_lines = content.split("\n")
    search_lines = [line.strip() for line in old_string.split("\n")]
    replace_lines = new_string.split("\n")

    occurrences = 0
    i = 0
    while i <= len(source_lines) - len(search_lines):
        window = source_lines[i : i + len(search_lines)]
        window_stripped = [line.strip() for line in window]

        if window_stripped == search_lines:
            occurrences += 1
            # Get indentation from first matching line
            indent_match = re.match(r"^(\s*)", window[0])
            indentation = indent_match.group(1) if indent_match else ""
            # Apply indentation to replacement
            indented_replace = [f"{indentation}{line}" for line in replace_lines]
            source_lines[i : i + len(search_lines)] = indented_replace
            i += len(replace_lines)
        else:
            i += 1

    return "\n".join(source_lines), occurrences


class GeminiEditTool(BaseTool):
    """Gemini CLI-style file editing tool (replace).

    Replaces text within a file. Uses three matching strategies:
    1. Exact string matching
    2. Flexible matching (whitespace-insensitive line comparison)
    3. Regex-based flexible matching

    By default replaces a single occurrence, but can replace multiple
    when expected_replacements is specified.

    Parameters (matching Gemini CLI exactly):
        file_path: Path to the file to modify (required)
        instruction: Semantic description of the change (required)
        old_string: Exact literal text to replace (required)
        new_string: Exact literal text to replace with (required)
        expected_replacements: Number of replacements expected (default: 1)

    Error messages match Gemini CLI format for consistency.

    Native specs: Uses function calling (no native API), but has role="editor"
                  for mutual exclusion with EditTool/ApplyPatchTool.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        # No api_type - uses standard function calling
        # Role ensures mutual exclusion with other editor tools
        AgentType.GEMINI: NativeToolSpec(role="editor"),
    }

    _base_directory: str
    _file_history: dict[Path, list[str]]

    def __init__(self, base_directory: str = ".") -> None:
        """Initialize GeminiEditTool.

        Args:
            base_directory: Base directory for relative paths
        """
        super().__init__(
            env=None,
            name="replace",  # Match Gemini CLI tool name
            title="Edit",
            description=(
                "Replaces text within a file. Requires providing significant context "
                "around the change. Always use read_file to examine content before editing. "
                "old_string MUST be exact literal text including whitespace and indentation. "
                "new_string MUST be exact literal text for the replacement."
            ),
        )
        self._base_directory = str(Path(base_directory).resolve())
        self._file_history = defaultdict(list)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to base directory."""
        path = Path(file_path)
        if path.is_absolute():
            return path
        return Path(self._base_directory) / path

    async def __call__(
        self,
        file_path: str,
        instruction: str,
        old_string: str,
        new_string: str,
        expected_replacements: int = 1,
    ) -> list[ContentBlock]:
        """Edit a file by replacing text.

        Uses three matching strategies (like Gemini CLI):
        1. Exact string matching
        2. Flexible matching (whitespace-insensitive)
        3. Regex-based flexible matching

        Args:
            file_path: Path to the file to modify
            instruction: Clear description of the change purpose
            old_string: Exact literal text to replace
            new_string: Exact literal text to replace with
            expected_replacements: Number of replacements expected (default: 1)

        Returns:
            List of ContentBlocks with Gemini CLI-style result
        """
        if not file_path:
            raise ToolError("The 'file_path' parameter must be non-empty.")
        if not instruction:
            raise ToolError("The 'instruction' parameter must be non-empty.")
        if old_string is None:
            raise ToolError("The 'old_string' parameter is required.")
        if new_string is None:
            raise ToolError("The 'new_string' parameter is required.")
        if expected_replacements < 1:
            raise ToolError("expected_replacements must be >= 1")

        path = self._resolve_path(file_path)

        if not path.exists():
            # Match Gemini CLI error format
            raise ToolError(f"File not found: {file_path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {file_path}")

        # Read current content
        file_content = read_file_sync(path)
        original_content = file_content

        # Normalize line endings and tabs
        file_content = file_content.replace("\r\n", "\n").expandtabs()
        old_string_norm = old_string.replace("\r\n", "\n").expandtabs()
        new_string_norm = new_string.replace("\r\n", "\n").expandtabs()

        # Strategy 1: Exact matching
        occurrences = file_content.count(old_string_norm)
        new_content = None
        match_strategy = "exact"

        if occurrences > 0:
            if occurrences == expected_replacements:
                new_content = file_content.replace(old_string_norm, new_string_norm)
            elif occurrences == 1 and expected_replacements == 1:
                new_content = file_content.replace(old_string_norm, new_string_norm, 1)

        # Strategy 2: Flexible matching (whitespace-insensitive)
        if new_content is None:
            flex_content, flex_occurrences = _flexible_match(
                file_content, old_string_norm, new_string_norm
            )
            if flex_occurrences == expected_replacements:
                new_content = flex_content
                occurrences = flex_occurrences
                match_strategy = "flexible"

        # Strategy 3: Regex-based flexible matching (for single replacements)
        if new_content is None and expected_replacements == 1:
            # Build flexible regex pattern
            tokens = old_string_norm.split()
            if tokens:
                escaped_tokens = [_escape_regex(t) for t in tokens]
                pattern = r"\s*".join(escaped_tokens)
                regex_match = re.search(pattern, file_content, re.MULTILINE)
                if regex_match:
                    new_content = (
                        file_content[: regex_match.start()]
                        + new_string_norm
                        + file_content[regex_match.end() :]
                    )
                    occurrences = 1
                    match_strategy = "regex"

        # Handle no match found
        if new_content is None or occurrences == 0:
            # Match Gemini CLI error format
            raise ToolError(
                f"Failed to edit, 0 occurrences found for old_string in {file_path}. "
                "Ensure you're not escaping content incorrectly and check whitespace, "
                "indentation, and context. Use read_file tool to verify."
            )

        # Handle occurrence count mismatch
        if occurrences != expected_replacements:
            # Find lines where occurrences appear
            lines_with_match = [
                idx + 1
                for idx, line in enumerate(file_content.split("\n"))
                if old_string_norm in line
            ]
            occurrence_term = "occurrence" if expected_replacements == 1 else "occurrences"
            raise ToolError(
                f"Failed to edit, Expected {expected_replacements} {occurrence_term} "
                f"but found {occurrences} for old_string in file: {file_path}"
            )

        # Check if old_string equals new_string
        if old_string_norm == new_string_norm:
            raise ToolError(
                f"No changes to apply. The old_string and new_string are identical "
                f"in file: {file_path}"
            )

        # Write new content
        write_file_sync(path, new_content)

        # Save to history for potential undo
        self._file_history[path].append(original_content)

        # Create snippet around the edit
        replacement_line = (
            file_content.split(old_string_norm)[0].count("\n")
            if old_string_norm in file_content
            else 0
        )
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_string_norm.count("\n")
        snippet = "\n".join(new_content.split("\n")[start_line : end_line + 1])

        # Build Gemini CLI-style success response
        result = f"Successfully modified file: {file_path} ({occurrences} replacements)."
        if match_strategy != "exact":
            result += f" [matched using {match_strategy} strategy]"

        return ContentResult(output=result).to_content_blocks()


__all__ = ["GeminiEditTool"]
