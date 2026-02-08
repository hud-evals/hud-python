"""Pluggable format conversion system for HUD.

Converts external benchmark formats (Harbor, Inspect AI, etc.) into
HUD environments + tasksets.

Usage:
    hud convert <path>                    # Auto-detect format
    hud convert <path> --from harbor      # Explicit format
    hud convert <path> --output ./out     # Custom output directory
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path  # noqa: TC003 - used at runtime

from .base import BaseConverter, ConvertResult, GeneratedEnvironment

__all__ = [
    "BaseConverter",
    "ConvertResult",
    "GeneratedEnvironment",
    "detect_format",
    "get_converter",
    "list_formats",
    "write_result",
]

LOGGER = logging.getLogger(__name__)

# Shell script extensions that need CRLF -> LF normalization
_SHELL_EXTENSIONS = frozenset({".sh", ".bash", ".zsh", ".ksh"})


def _normalize_line_endings(directory: Path) -> None:
    """Convert CRLF to LF in all shell scripts under a directory.

    Git on Windows with autocrlf=true converts LF to CRLF on checkout.
    Shell scripts with CRLF break on Linux (e.g., shebang errors,
    'set: pipefail\\r: invalid option name').
    """
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in _SHELL_EXTENSIONS:
            raw = path.read_bytes()
            if b"\r" in raw:
                path.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
                LOGGER.debug("Normalized line endings: %s", path)


# ---------------------------------------------------------------------------
# Converter registry
# ---------------------------------------------------------------------------

# Lazy-loaded to avoid import cost on unrelated CLI commands
_converters: list[BaseConverter] | None = None


def _load_converters() -> list[BaseConverter]:
    global _converters
    if _converters is None:
        from .harbor import HarborConverter

        _converters = [
            HarborConverter(),
            # Future: InspectConverter(), METRConverter(), ...
        ]
    return _converters


def get_converter(name: str) -> BaseConverter | None:
    """Get a converter by its short name (e.g., 'harbor')."""
    for c in _load_converters():
        if c.name == name:
            return c
    return None


def detect_format(path: Path) -> BaseConverter | None:
    """Auto-detect which converter can handle the given path."""
    for c in _load_converters():
        if c.detect(path):
            return c
    return None


def list_formats() -> list[tuple[str, str]]:
    """Return (name, description) pairs for all registered converters."""
    return [(c.name, c.description) for c in _load_converters()]


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def write_result(result: ConvertResult, output_dir: Path) -> Path:
    """Write conversion results to disk.

    Creates the output directory structure:
        output_dir/
        ├── env-name-a/
        │   ├── env.py
        │   ├── Dockerfile.hud
        │   ├── pyproject.toml
        │   └── tasks/
        │       └── <task_id>/  (copied from source, minus environment/ & solution/)
        └── taskset.json

    Returns the path to the generated taskset.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_gen in result.environments:
        env_dir = output_dir / env_gen.name
        env_dir.mkdir(parents=True, exist_ok=True)

        # Write generated files
        (env_dir / "env.py").write_text(env_gen.env_py, encoding="utf-8")
        (env_dir / "Dockerfile.hud").write_text(env_gen.dockerfile, encoding="utf-8")
        (env_dir / "pyproject.toml").write_text(env_gen.pyproject_toml, encoding="utf-8")

        # Copy build context files from source environment/ directory
        # (e.g., warriors/*.red that Harbor Dockerfiles reference via COPY)
        if env_gen.build_context_source and env_gen.build_context_source.is_dir():
            for item in env_gen.build_context_source.iterdir():
                # Skip the Dockerfile itself (we already generated Dockerfile.hud)
                if item.name.lower() in ("dockerfile", "dockerfile.hud"):
                    continue
                dest_item = env_dir / item.name
                if dest_item.exists():
                    if dest_item.is_dir():
                        shutil.rmtree(dest_item)
                    else:
                        dest_item.unlink()
                if item.is_dir():
                    shutil.copytree(item, dest_item)
                else:
                    shutil.copy2(item, dest_item)

        # Copy task data directories (skip environment/ and solution/)
        tasks_dir = env_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)

        for task_id, source_dir in env_gen.task_dirs.items():
            dest = tasks_dir / task_id
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)

            for item in source_dir.iterdir():
                # Skip dirs that are handled by the Dockerfile or ignored
                if item.name in ("environment", "solution"):
                    continue
                if item.is_dir():
                    shutil.copytree(item, dest / item.name)
                else:
                    shutil.copy2(item, dest / item.name)

        # Normalize CRLF -> LF in all shell scripts (fixes Windows git checkout)
        _normalize_line_endings(env_dir)

        LOGGER.info(
            "Wrote environment '%s' with %d task(s)",
            env_gen.name,
            len(env_gen.task_dirs),
        )

    # Write taskset
    taskset_path = output_dir / "taskset.json"
    with open(taskset_path, "w", encoding="utf-8") as f:
        json.dump(result.taskset, f, ensure_ascii=False, indent=2)
        f.write("\n")

    LOGGER.info("Wrote taskset with %d task(s) to %s", len(result.taskset), taskset_path)
    return taskset_path
