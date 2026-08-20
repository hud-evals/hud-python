"""Build context tarball creation for the deploy command."""

from __future__ import annotations

import fnmatch
import gzip
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from hud.build_context import BuildContextManifest
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

    with ignore_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)

    return patterns


def _matches_pattern(
    rel_path_str: str,
    path: Path,
    pattern: str,
) -> bool:
    """Check if a path matches a single ignore pattern.

    Args:
        rel_path_str: Relative path string (forward slashes)
        path: Original Path object (for is_dir checks)
        pattern: Single ignore pattern (without ! prefix)

    Returns:
        True if the path matches the pattern
    """
    # Handle directory-only patterns (ending with /)
    if pattern.endswith("/"):
        pattern = pattern[:-1]
        if path.is_dir() and fnmatch.fnmatch(rel_path_str, pattern):
            return True
        return fnmatch.fnmatch(rel_path_str, f"{pattern}/*")

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
        return False

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


def should_ignore(
    path: Path,
    base_path: Path,
    ignore_patterns: list[str],
) -> bool:
    """Check if a path should be ignored based on patterns.

    Supports negation patterns (lines starting with !) following
    .dockerignore semantics: patterns are evaluated in order, and
    a later negation can re-include a previously excluded path.

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

    ignored = False

    for pattern in ignore_patterns:
        # Handle negation patterns
        if pattern.startswith("!"):
            # A negation pattern re-includes a previously excluded file
            neg_pattern = pattern[1:]
            if ignored and _matches_pattern(rel_path_str, path, neg_pattern):
                ignored = False
        elif _matches_pattern(rel_path_str, path, pattern):
            ignored = True

    return ignored


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


@dataclass(frozen=True, slots=True)
class BuildContextArchive:
    path: Path
    manifest: BuildContextManifest
    size_bytes: int
    file_count: int
    duration_seconds: float


def _build_context_paths(
    directory: Path,
    dockerignore_path: Path | None = None,
    verbose: bool = False,
) -> list[Path]:
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

    paths: list[Path] = []
    for root, dirs, files in os.walk(directory):
        root_path = Path(root)
        retained_dirs: list[str] = []
        for name in sorted(dirs):
            path = root_path / name
            if should_ignore(path, directory, ignore_patterns):
                if verbose:
                    hud_console.debug(f"Skipping: {path.relative_to(directory)}")
                continue
            if path.is_symlink():
                paths.append(path)
            else:
                retained_dirs.append(name)
        dirs[:] = retained_dirs

        for name in sorted(files):
            path = root_path / name
            if should_ignore(path, directory, ignore_patterns):
                if verbose:
                    hud_console.debug(f"Skipping: {path.relative_to(directory)}")
                continue
            paths.append(path)

    return paths


def create_build_context_tarball(
    directory: Path,
    dockerignore_path: Path | None = None,
    verbose: bool = False,
) -> BuildContextArchive:
    """Create a tarball and canonical manifest from one selected file set."""
    start_time = time.time()
    directory = directory.resolve()
    manifest = BuildContextManifest.from_paths(
        directory,
        _build_context_paths(directory, dockerignore_path, verbose),
    )
    temp_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
        suffix=".tar.gz",
        delete=False,
        prefix="hud-build-context-",
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        with (
            temp_path.open("wb") as raw_archive,
            gzip.GzipFile(
                filename="",
                fileobj=raw_archive,
                mode="wb",
                mtime=0,
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as tar,
        ):
            for entry in manifest.entries:
                info = tarfile.TarInfo(entry.path)
                info.mode = entry.mode
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                if entry.type == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = entry.target or ""
                    tar.addfile(info)
                else:
                    info.size = entry.size or 0
                    with (directory / entry.path).open("rb") as source:
                        tar.addfile(info, source)

        size_bytes = temp_path.stat().st_size
        duration = time.time() - start_time
        return BuildContextArchive(
            path=temp_path,
            manifest=manifest,
            size_bytes=size_bytes,
            file_count=len(manifest.entries),
            duration_seconds=duration,
        )

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
