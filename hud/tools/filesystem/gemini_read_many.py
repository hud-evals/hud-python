"""Gemini CLI-style read_many_files tool.

Based on Gemini CLI's read_many_files tool:
https://github.com/google-gemini/gemini-cli

Reads content from multiple files specified by glob patterns,
concatenating results with file path separators.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

from mcp.types import TextContent  # noqa: TC002

from hud.tools.filesystem.base import BaseFilesystemTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.tools.types import ContentResult, ToolError
from hud.types import AgentType


class GeminiReadManyTool(BaseFilesystemTool):
    """Gemini CLI-style multi-file reading tool.

    Reads content from multiple files specified by glob patterns or paths.
    Concatenates results with file path separators.

    Parameters (matching Gemini CLI):
        include: Array of glob patterns or file paths to read (required)
        exclude: Array of glob patterns to exclude (optional)
        recursive: Whether to search recursively (default: True)
        useDefaultExcludes: Whether to apply default exclusions (default: True)
    """

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.GEMINI: NativeToolSpec(role="batch_reader"),
    }

    _max_files: int
    _max_total_lines: int

    def __init__(
        self,
        base_path: str = ".",
        max_files: int = 100,
        max_total_lines: int = 10000,
    ) -> None:
        super().__init__(
            base_path=base_path,
            name="read_many_files",
            title="ReadManyFiles",
            description=(
                "Reads and concatenates content from multiple files. "
                "Accepts arrays of glob patterns and file paths to include and exclude. "
                "Returns concatenated file contents separated by file path headers."
            ),
        )
        self._max_files = max_files
        self._max_total_lines = max_total_lines

    def _collect_files(
        self,
        include: list[str],
        exclude: list[str] | None,
        recursive: bool,
        use_default_excludes: bool,
    ) -> list[Path]:
        """Resolve include/exclude patterns into a deduplicated file list."""
        exclude_patterns = exclude or []
        seen: set[Path] = set()
        files: list[Path] = []

        for pattern in include:
            resolved = self.resolve_path(pattern)

            # Literal file path
            if resolved.is_file():
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)
                continue

            # Glob pattern
            base = self._base_path
            if recursive and "**" not in pattern:
                pattern = f"**/{pattern}"

            for match in base.glob(pattern):
                if not match.is_file():
                    continue
                if match in seen:
                    continue
                if self.is_hidden(match):
                    continue
                if use_default_excludes and self.is_ignored_dir(match):
                    continue
                seen.add(match)
                files.append(match)

                if len(files) >= self._max_files:
                    break

            if len(files) >= self._max_files:
                break

        # Apply exclude patterns
        if exclude_patterns:
            files = [
                f
                for f in files
                if not any(fnmatch.fnmatch(str(f), ep) for ep in exclude_patterns)
                and not any(fnmatch.fnmatch(f.name, ep) for ep in exclude_patterns)
            ]

        return files

    async def __call__(
        self,
        include: list[str],
        exclude: list[str] | None = None,
        recursive: bool = True,
        useDefaultExcludes: bool = True,
    ) -> list[TextContent]:
        """Read content from multiple files.

        Args:
            include: Array of glob patterns or file paths to read
            exclude: Array of glob patterns to exclude
            recursive: Whether to search recursively
            useDefaultExcludes: Whether to apply default exclusions

        Returns:
            List of TextContent with concatenated file contents
        """
        if not include:
            raise ToolError("The 'include' parameter must be a non-empty array.")

        files = self._collect_files(include, exclude, recursive, useDefaultExcludes)

        if not files:
            return ContentResult(
                output="No files found matching the specified patterns."
            ).to_text_blocks()

        parts: list[str] = []
        total_lines = 0
        files_read = 0
        skipped: list[str] = []
        truncated = False

        for path in files:
            try:
                content = self.read_file_content(path)
            except Exception:
                skipped.append(str(path))
                continue

            line_count = content.count("\n") + 1
            if total_lines + line_count > self._max_total_lines:
                truncated = True
                remaining = self._max_total_lines - total_lines
                if remaining > 0:
                    truncated_content = "\n".join(content.split("\n")[:remaining])
                    try:
                        rel = str(path.relative_to(self._base_path))
                    except ValueError:
                        rel = str(path)
                    parts.append(f"--- {rel} ---")
                    parts.append(truncated_content)
                    parts.append(
                        "[WARNING: This file was truncated. Use 'read_file' for full content.]"
                    )
                    files_read += 1
                break

            try:
                rel = str(path.relative_to(self._base_path))
            except ValueError:
                rel = str(path)

            parts.append(f"--- {rel} ---")
            parts.append(content)
            total_lines += line_count
            files_read += 1

        parts.append("--- End of content ---")

        if skipped:
            parts.append(f"\nSkipped {len(skipped)} files (read errors): {', '.join(skipped)}")

        if truncated:
            parts.append(
                f"\n[Truncated: showing {files_read} of {len(files)} files, "
                f"{total_lines} lines. Use more specific patterns or "
                f"read_file for individual files.]"
            )

        output = "\n".join(parts)
        return ContentResult(output=output).to_text_blocks()


__all__ = ["GeminiReadManyTool"]
