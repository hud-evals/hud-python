"""Build context tarball creation for direct deploys."""

from __future__ import annotations

import fnmatch
import os
import tarfile
import tempfile
import time
from pathlib import Path

from hud.utils.hud_console import HUDConsole


def parse_ignore_file(ignore_path: Path) -> list[str]:
    """Parse a .dockerignore or .gitignore file and return a list of patterns.

    Args:
        ignore_path: Path to the ignore file (.dockerignore or .gitignore)

    Returns:
        List of ignore patterns
    """
    patterns: list[str] = []
    if not ignore_path.exists():
        return patterns

    try:
        with open(ignore_path) as f:
            for line in f:
                # Strip whitespace and skip comments/empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
    except Exception:  # noqa: S110
        pass  # Best effort - ignore parse errors

    return patterns


def should_ignore(
    path: Path,
    base_path: Path,
    ignore_patterns: list[str],
) -> bool:
    """Check if a path should be ignored based on patterns.

    Args:
        path: Path to check
        base_path: Base directory for relative path calculation
        ignore_patterns: List of ignore patterns

    Returns:
        True if the path should be ignored
    """
    try:
        rel_path = path.relative_to(base_path)
        rel_path_str = str(rel_path).replace("\\", "/")
    except ValueError:
        return False

    for pattern in ignore_patterns:
        # Handle negation patterns (we don't support these yet, just skip)
        if pattern.startswith("!"):
            continue

        # Handle directory-only patterns (ending with /)
        if pattern.endswith("/"):
            pattern = pattern[:-1]
            if path.is_dir() and fnmatch.fnmatch(rel_path_str, pattern):
                return True
            if fnmatch.fnmatch(rel_path_str, f"{pattern}/*"):
                return True
            continue

        # Handle ** patterns (match any directory depth)
        if "**" in pattern:
            # Convert ** to regex-like pattern
            regex_pattern = pattern.replace("**", "*")
            if fnmatch.fnmatch(rel_path_str, regex_pattern):
                return True
            # Also check if any parent directory matches
            parts = rel_path_str.split("/")
            for i in range(len(parts)):
                partial = "/".join(parts[: i + 1])
                if fnmatch.fnmatch(partial, regex_pattern):
                    return True
            continue

        # Standard pattern matching
        if fnmatch.fnmatch(rel_path_str, pattern):
            return True
        # Also match against just the filename
        if fnmatch.fnmatch(path.name, pattern):
            return True
        # Check if pattern matches a parent directory
        parts = rel_path_str.split("/")
        for i in range(len(parts)):
            partial = "/".join(parts[: i + 1])
            if fnmatch.fnmatch(partial, pattern):
                return True

    return False


# Default patterns that are always excluded for security and efficiency
DEFAULT_EXCLUDES = [
    ".git",
    ".git/*",
    "__pycache__",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    ".env",  # Never include secrets!
    ".env.*",
    "*.env",
    ".venv",
    ".venv/*",
    "venv",
    "venv/*",
    "node_modules",
    "node_modules/*",
    ".mypy_cache",
    ".mypy_cache/*",
    ".pytest_cache",
    ".pytest_cache/*",
    ".ruff_cache",
    ".ruff_cache/*",
    "*.egg-info",
    "*.egg-info/*",
    "dist",
    "dist/*",
    "build",
    "build/*",
    ".DS_Store",
    "Thumbs.db",
]


def create_build_context_tarball(
    directory: Path,
    dockerignore_path: Path | None = None,
    verbose: bool = False,
) -> tuple[Path, int, int, float]:
    """Create a gzipped tarball of the build context.

    Respects .dockerignore and .gitignore patterns, and always excludes
    common sensitive files like .env and .git directories.

    Args:
        directory: Directory to create tarball from
        dockerignore_path: Optional path to .dockerignore file.
                          If None, looks for .dockerignore in directory.
        verbose: Whether to print verbose output

    Returns:
        Tuple of (tarball_path, size_bytes, file_count, duration_seconds)
    """
    start_time = time.time()
    hud_console = HUDConsole()
    directory = directory.resolve()

    # Build ignore patterns from multiple sources
    ignore_patterns = list(DEFAULT_EXCLUDES)
    loaded_sources: list[str] = []

    # Add patterns from .gitignore (read first, lower priority)
    gitignore_path = directory / ".gitignore"
    if gitignore_path.exists():
        gitignore_patterns = parse_ignore_file(gitignore_path)
        ignore_patterns.extend(gitignore_patterns)
        loaded_sources.append(f".gitignore ({len(gitignore_patterns)} patterns)")

    # Add patterns from .dockerignore (read second, higher priority)
    if dockerignore_path is None:
        dockerignore_path = directory / ".dockerignore"
    if dockerignore_path.exists():
        dockerignore_patterns = parse_ignore_file(dockerignore_path)
        ignore_patterns.extend(dockerignore_patterns)
        loaded_sources.append(f".dockerignore ({len(dockerignore_patterns)} patterns)")

    if verbose and loaded_sources:
        hud_console.info(f"Loaded ignore patterns from: {', '.join(loaded_sources)}")

    # Create temporary file for tarball
    temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        suffix=".tar.gz",
        delete=False,
        prefix="hud-build-context-",
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    file_count = 0

    try:
        with tarfile.open(temp_path, "w:gz") as tar:
            for root, dirs, files in os.walk(directory):
                root_path = Path(root)

                # Filter directories in-place to skip ignored ones
                dirs[:] = [
                    d for d in dirs if not should_ignore(root_path / d, directory, ignore_patterns)
                ]

                for file in files:
                    file_path = root_path / file

                    if should_ignore(file_path, directory, ignore_patterns):
                        if verbose:
                            hud_console.debug(f"Skipping: {file_path.relative_to(directory)}")
                        continue

                    # Add file to tarball with relative path
                    arcname = str(file_path.relative_to(directory))
                    tar.add(file_path, arcname=arcname)
                    file_count += 1

                    if verbose:
                        hud_console.debug(f"Added: {arcname}")

        size_bytes = temp_path.stat().st_size
        duration = time.time() - start_time
        return temp_path, size_bytes, file_count, duration

    except Exception:
        # Clean up temp file on error
        temp_path.unlink(missing_ok=True)
        raise


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "1.5 MB")
    """
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
