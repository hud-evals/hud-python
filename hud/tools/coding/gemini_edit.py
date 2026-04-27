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
    read_file_sync,
    write_file_sync,
)


def _escape_regex(s: str) -> str:
    """Escape regex special characters."""
    return re.sub(r"[.*+?^${}()|[\]\\]", r"\\\g<0>", s)


def _tokenize_for_regex(s: str) -> list[str]:
    """Tokenize string by splitting on delimiters (matching Gemini CLI).

    Pads delimiters with spaces before splitting so each delimiter
    becomes its own token. E.g., "foo(bar)" -> ["foo", "(", "bar", ")"].
    """
    processed = s
    for delim in "():[]{}><= ":
        processed = processed.replace(delim, f" {delim} ")
    return [t for t in processed.split() if t]


def _detect_line_ending(content: str) -> str:
    """Detect the dominant line ending in content."""
    crlf = content.count("\r\n")
    lf = content.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _restore_trailing_newline(new_content: str, original_content: str) -> str:
    """Preserve the original file's trailing newline state."""
    had_trailing = original_content.endswith("\n")
    has_trailing = new_content.endswith("\n")
    if had_trailing and not has_trailing:
        return new_content + "\n"
    if not had_trailing and has_trailing:
        return new_content.rstrip("\n")
    return new_content


def _apply_relative_indentation(
    base_indent: str,
    old_lines: list[str],
    new_lines: list[str],
) -> list[str]:
    """Apply indentation preserving relative indent levels.

    Uses the first old line's indent as reference, computes each
    new line's relative indent, then applies base_indent + relative.
    """
    if not new_lines:
        return new_lines

    # Determine reference indent from old_lines
    if old_lines:
        ref_match = re.match(r"^(\s*)", old_lines[0])
        ref_indent = ref_match.group(1) if ref_match else ""
    else:
        ref_indent = ""

    result = []
    for j, line in enumerate(new_lines):
        if not line.strip():
            result.append("")
            continue
        if j == 0:
            result.append(f"{base_indent}{line.lstrip()}")
        else:
            line_match = re.match(r"^(\s*)", line)
            line_indent = line_match.group(1) if line_match else ""
            extra = line_indent[len(ref_indent) :] if len(line_indent) > len(ref_indent) else ""
            result.append(f"{base_indent}{extra}{line.lstrip()}")
    return result


def _flexible_match(content: str, old_string: str, new_string: str) -> tuple[str, int]:
    """Attempt flexible whitespace-insensitive matching.

    Matches Gemini CLI behavior: strips each line and compares,
    preserves relative indentation in replacement.
    """
    source_lines = content.split("\n")
    search_lines = [line.strip() for line in old_string.split("\n")]
    replace_lines = new_string.split("\n")
    old_lines = old_string.split("\n")

    occurrences = 0
    i = 0
    while i <= len(source_lines) - len(search_lines):
        window = source_lines[i : i + len(search_lines)]
        window_stripped = [line.strip() for line in window]

        if window_stripped == search_lines:
            occurrences += 1
            indent_match = re.match(r"^(\s*)", window[0])
            base_indent = indent_match.group(1) if indent_match else ""
            indented = _apply_relative_indentation(base_indent, old_lines, replace_lines)
            source_lines[i : i + len(search_lines)] = indented
            i += len(indented)
        else:
            i += 1

    return "\n".join(source_lines), occurrences


class GeminiEditTool(BaseTool):
    """Gemini CLI-style file editing tool (replace).

    Replaces text within a file. Uses three matching strategies:
    1. Exact string matching
    2. Flexible matching (whitespace-insensitive line comparison)
    3. Regex-based flexible matching

    When old_string is empty and the file does not exist, creates a new file
    with new_string as content.

    Parameters (matching Gemini CLI exactly):
        file_path: Path to the file to modify (required)
        instruction: Semantic description of the change (required)
        old_string: Exact literal text to replace (required)
        new_string: Exact literal text to replace with (required)
        allow_multiple: If true, replace all occurrences (default: false)

    Native specs: Uses function calling (no native API), but has role="editor"
                  for mutual exclusion with EditTool/ApplyPatchTool.
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(role="editor"),
    }

    _base_directory: str
    _file_history: dict[Path, list[str]]

    def __init__(self, base_directory: str = ".") -> None:
        super().__init__(
            env=None,
            name="replace",
            title="Edit",
            description=(
                "Replaces text within a file. Requires providing significant context "
                "around the change. Always use read_file to examine content before editing. "
                "old_string MUST be exact literal text including whitespace and indentation. "
                "new_string MUST be exact literal text for the replacement. "
                "To create a new file, set old_string to empty string."
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
        allow_multiple: bool = False,
    ) -> list[ContentBlock]:
        """Edit a file by replacing text, or create a new file.

        Args:
            file_path: Path to the file to modify
            instruction: Clear description of the change purpose
            old_string: Exact literal text to replace (empty = create file)
            new_string: Exact literal text to replace with
            allow_multiple: If true, replace all occurrences (default: false)

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

        path = self._resolve_path(file_path)

        # File creation: empty old_string on non-existent file
        if old_string == "" and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_file_sync(path, new_string)
            return ContentResult(output=f"Created new file: {file_path}").to_content_blocks()

        if old_string == "" and path.exists():
            raise ToolError(
                f"File already exists, cannot create: {file_path}. "
                "Use a non-empty old_string to edit an existing file."
            )

        if not path.exists():
            raise ToolError(f"File not found: {file_path}")
        if path.is_dir():
            raise ToolError(f"Path is a directory: {file_path}")

        # Read current content
        file_content = read_file_sync(path)
        original_content = file_content

        # Detect and normalize line endings (restore later)
        original_ending = _detect_line_ending(file_content)
        file_content = file_content.replace("\r\n", "\n")
        old_string_norm = old_string.replace("\r\n", "\n")
        new_string_norm = new_string.replace("\r\n", "\n")

        # Strategy 1: Exact matching
        occurrences = file_content.count(old_string_norm)
        new_content = None
        match_strategy = "exact"

        if occurrences > 0:
            if allow_multiple:
                new_content = file_content.replace(old_string_norm, new_string_norm)
            elif occurrences == 1:
                new_content = file_content.replace(old_string_norm, new_string_norm, 1)
            else:
                raise ToolError(
                    f"Multiple occurrences ({occurrences}) found for "
                    f"old_string in {file_path}. "
                    "Use allow_multiple: true to replace all, or provide "
                    "more context to match a single occurrence."
                )

        # Strategy 2: Flexible matching (whitespace-insensitive)
        if new_content is None:
            flex_content, flex_occurrences = _flexible_match(
                file_content, old_string_norm, new_string_norm
            )
            if flex_occurrences > 0:
                if allow_multiple or flex_occurrences == 1:
                    new_content = flex_content
                    occurrences = flex_occurrences
                    match_strategy = "flexible"
                else:
                    raise ToolError(
                        f"Multiple occurrences ({flex_occurrences}) found "
                        f"for old_string in {file_path}. "
                        "Use allow_multiple: true to replace all."
                    )

        # Strategy 3: Regex-based flexible matching
        if new_content is None:
            tokens = _tokenize_for_regex(old_string_norm)
            if tokens:
                escaped_tokens = [_escape_regex(t) for t in tokens]
                pattern = r"^([ \t]*)" + r"\s*".join(escaped_tokens)
                if allow_multiple:
                    regex_matches = list(re.finditer(pattern, file_content, re.MULTILINE))
                    if regex_matches:
                        # Replace from end to start to preserve offsets
                        new_content = file_content
                        for m in reversed(regex_matches):
                            new_content = (
                                new_content[: m.start()]
                                + m.group(1)
                                + new_string_norm
                                + new_content[m.end() :]
                            )
                        occurrences = len(regex_matches)
                        match_strategy = "regex"
                else:
                    regex_match = re.search(pattern, file_content, re.MULTILINE)
                    if regex_match:
                        indent = regex_match.group(1)
                        new_content = (
                            file_content[: regex_match.start()]
                            + indent
                            + new_string_norm
                            + file_content[regex_match.end() :]
                        )
                        occurrences = 1
                        match_strategy = "regex"

        # Handle no match found
        if new_content is None or occurrences == 0:
            raise ToolError(
                f"Failed to edit, 0 occurrences found for old_string "
                f"in {file_path}. "
                "Ensure you're not escaping content incorrectly and "
                "check whitespace, indentation, and context. "
                "Use read_file tool to verify."
            )

        # Check if old_string equals new_string
        if old_string_norm == new_string_norm:
            raise ToolError(
                "No changes to apply. The old_string and new_string "
                f"are identical in file: {file_path}"
            )

        # Restore trailing newline state and line endings
        new_content = _restore_trailing_newline(new_content, file_content)
        if original_ending == "\r\n":
            new_content = new_content.replace("\n", "\r\n")

        # Write new content
        write_file_sync(path, new_content)

        # Save to history for potential undo
        self._file_history[path].append(original_content)

        result = f"Successfully modified file: {file_path} ({occurrences} replacements)."
        if match_strategy != "exact":
            result += f" [matched using {match_strategy} strategy]"

        return ContentResult(output=result).to_content_blocks()


__all__ = ["GeminiEditTool"]
