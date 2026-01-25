"""Shared utilities for coding tools.

Common file I/O, snippet generation, and path handling used by
shell, bash, edit, and apply_patch tools.
"""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

from hud.tools.types import ToolError

# Default number of lines to show around edits in snippets
SNIPPET_LINES: int = 4

# Maximum content length before truncation
MAX_RESPONSE_LENGTH: int = 16000


def maybe_truncate(content: str, max_length: int = MAX_RESPONSE_LENGTH) -> str:
    """Truncate content if it exceeds max length."""
    if len(content) <= max_length:
        return content
    half = max_length // 2
    return content[:half] + "\n\n... [truncated] ...\n\n" + content[-half:]


def make_snippet(
    content: str,
    descriptor: str,
    start_line: int = 1,
    expand_tabs: bool = True,
) -> str:
    """Generate a snippet of file content with line numbers.

    Args:
        content: File content to display
        descriptor: Description of the content (e.g., file path)
        start_line: Starting line number for numbering
        expand_tabs: Whether to expand tabs to spaces

    Returns:
        Formatted snippet with line numbers
    """
    content = maybe_truncate(content)
    if expand_tabs:
        content = content.expandtabs()
    lines = content.split("\n")
    numbered = [f"{i + start_line:6}\t{line}" for i, line in enumerate(lines)]
    return f"Here's the result of running `cat -n` on {descriptor}:\n" + "\n".join(numbered) + "\n"


async def read_file_async(path: Path) -> str:
    """Read file content asynchronously using subprocess (for sandboxed environments).

    Args:
        path: Path to the file to read

    Returns:
        File content as string

    Raises:
        ToolError: If file cannot be read
    """
    try:
        safe_path = shlex.quote(str(path))
        process = await asyncio.create_subprocess_shell(
            f"cat {safe_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise ToolError(f"Failed to read {path}: {stderr.decode()}")
        return stdout.decode()
    except Exception as e:
        raise ToolError(f"Failed to read {path}: {e}") from None


async def write_file_async(path: Path, content: str) -> None:
    """Write file content asynchronously using subprocess (for sandboxed environments).

    Args:
        path: Path to the file to write
        content: Content to write

    Raises:
        ToolError: If file cannot be written
    """
    try:
        safe_path = shlex.quote(str(path))
        process = await asyncio.create_subprocess_shell(
            f"cat > {safe_path} << 'EOF'\n{content}\nEOF",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise ToolError(f"Failed to write {path}: {stderr.decode()}")
    except Exception as e:
        raise ToolError(f"Failed to write {path}: {e}") from None


def read_file_sync(path: Path) -> str:
    """Read file content synchronously (for local environments).

    Args:
        path: Path to the file to read

    Returns:
        File content as string

    Raises:
        ToolError: If file cannot be read
    """
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise ToolError(f"Failed to read {path}: {e}") from None


def write_file_sync(path: Path, content: str) -> None:
    """Write file content synchronously (for local environments).

    Args:
        path: Path to the file to write
        content: Content to write

    Raises:
        ToolError: If file cannot be written
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        raise ToolError(f"Failed to write {path}: {e}") from None


def validate_path(path: Path, must_exist: bool = True, allow_dir: bool = False) -> None:
    """Validate a file path.

    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        allow_dir: Whether directories are allowed

    Raises:
        ToolError: If validation fails
    """
    if not path.is_absolute():
        raise ToolError(f"Path {path} is not absolute. Use an absolute path starting with '/'.")
    if must_exist and not path.exists():
        raise ToolError(f"Path {path} does not exist.")
    if path.exists() and path.is_dir() and not allow_dir:
        raise ToolError(f"Path {path} is a directory, expected a file.")


def resolve_path_safely(file_path: str, base_path: Path) -> Path:
    """Resolve a file path, ensuring it stays within base_path.

    Used by filesystem tools (read, grep, glob, list) for security.

    Args:
        file_path: The path to resolve (relative or absolute)
        base_path: The base directory that must contain the result

    Returns:
        Resolved absolute Path

    Raises:
        ToolError: If path escapes base directory
    """
    path = Path(file_path)
    resolved = path.resolve() if path.is_absolute() else (base_path / path).resolve()

    # Security: ensure path is within base_path
    try:
        resolved.relative_to(base_path)
    except ValueError:
        raise ToolError(f"Path escapes base directory: {file_path}") from None

    return resolved


__all__ = [
    "MAX_RESPONSE_LENGTH",
    "SNIPPET_LINES",
    "make_snippet",
    "maybe_truncate",
    "read_file_async",
    "read_file_sync",
    "resolve_path_safely",
    "validate_path",
    "write_file_async",
    "write_file_sync",
]
