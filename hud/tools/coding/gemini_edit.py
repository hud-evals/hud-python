"""Gemini-style edit tool implementation.

Based on Gemini CLI's edit tool:
https://github.com/google-gemini/gemini-cli

This provides a simpler str_replace interface compared to Claude's EditTool,
matching the Gemini CLI parameter names and behavior.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import ClassVar

from mcp.types import ContentBlock

from hud.tools.base import BaseTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType

from .utils import (
    SNIPPET_LINES,
    make_snippet,
    maybe_truncate,
    read_file_sync,
    write_file_sync,
)


class GeminiEditTool(BaseTool):
    """Gemini CLI-style file editing tool.

    Replaces text within a file using exact string matching.
    By default replaces a single occurrence, but can replace multiple
    when expected_replacements is specified.

    Parameters (matching Gemini CLI):
        file_path: Path to the file to modify (required)
        instruction: Semantic description of the change (required)
        old_string: Exact literal text to replace (required)
        new_string: Exact literal text to replace with (required)
        expected_replacements: Number of replacements expected (default: 1)

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
            name="edit",
            title="Edit File",
            description=(
                "Replace text within a file. Requires exact string matching. "
                "Include 3+ lines of context around the target text."
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

        Args:
            file_path: Path to the file to modify
            instruction: Clear description of the change purpose
            old_string: Exact literal text to replace
            new_string: Exact literal text to replace with
            expected_replacements: Number of replacements expected (default: 1)

        Returns:
            List of ContentBlocks with result
        """
        if not file_path:
            raise ToolError("file_path is required")
        if not instruction:
            raise ToolError("instruction is required")
        if old_string is None:
            raise ToolError("old_string is required")
        if new_string is None:
            raise ToolError("new_string is required")
        if expected_replacements < 1:
            raise ToolError("expected_replacements must be >= 1")

        path = self._resolve_path(file_path)

        if not path.exists():
            raise ToolError(f"File not found: {path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {path}")

        # Read current content (sync version for local filesystem)
        file_content = read_file_sync(path)
        original_content = file_content

        # Normalize tabs for matching
        file_content = file_content.expandtabs()
        old_string = old_string.expandtabs()
        new_string = new_string.expandtabs()

        # Count occurrences
        occurrences = file_content.count(old_string)

        if occurrences == 0:
            raise ToolError(
                f"old_string not found in {file_path}. "
                "Ensure the text matches exactly including whitespace and indentation."
            )

        if occurrences != expected_replacements:
            # Find lines where occurrences appear for better error message
            lines_with_match = []
            for idx, line in enumerate(file_content.split("\n")):
                if old_string in line:
                    lines_with_match.append(idx + 1)

            raise ToolError(
                f"Expected {expected_replacements} replacement(s) but found {occurrences} "
                f"occurrence(s) of old_string in {file_path} at lines {lines_with_match}. "
                "Include more context to make the match unique, or adjust expected_replacements."
            )

        # Perform replacement
        if expected_replacements == 1:
            # Replace only first occurrence
            new_content = file_content.replace(old_string, new_string, 1)
        else:
            # Replace all occurrences (already validated count matches)
            new_content = file_content.replace(old_string, new_string)

        # Write new content
        write_file_sync(path, new_content)

        # Save to history for potential undo
        self._file_history[path].append(original_content)

        # Create snippet around the edit
        replacement_line = file_content.split(old_string)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_string.count("\n")
        snippet = "\n".join(new_content.split("\n")[start_line : end_line + 1])

        # Build response
        result = (
            f"Successfully edited {file_path}\n"
            f"Instruction: {instruction}\n"
            f"Replacements: {expected_replacements}\n\n"
            f"{make_snippet(snippet, str(file_path), start_line + 1)}"
        )

        return ContentResult(output=result).to_content_blocks()


__all__ = ["GeminiEditTool"]
