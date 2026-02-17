"""Abstract base classes for format converters.

The converter system is pluggable: each format (Harbor, Inspect AI, etc.)
implements BaseConverter with detect() and convert() methods. The CLI
auto-detects format or lets the user specify explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["BaseConverter", "ConvertResult", "GeneratedEnvironment"]


class GeneratedEnvironment(BaseModel):
    """A generated HUD environment ready to be written to disk.

    Attributes:
        name: Environment name (e.g., "hud-harbor-algotune")
        env_py: Generated env.py file content
        dockerfile: Generated Dockerfile.hud content
        pyproject_toml: Generated pyproject.toml content
        task_dirs: Mapping of task_id -> source directory path.
            Files from these directories (minus environment/ and solution/)
            are copied into the output's tasks/ subdirectory.
        build_context_source: Optional path to a source directory whose
            non-Dockerfile contents should be copied into the environment
            root as Docker build context (e.g., Harbor's environment/ dir).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    env_py: str
    dockerfile: str
    pyproject_toml: str
    task_dirs: dict[str, Path]
    build_context_source: Path | None = None


class ConvertResult(BaseModel):
    """Result of converting a source format to HUD.

    Attributes:
        environments: Generated environment definitions (one per unique env group)
        taskset: List of v5 Task dicts ready for taskset.json
        summary: Human-readable summary lines for CLI output
    """

    environments: list[GeneratedEnvironment]
    taskset: list[dict[str, Any]]
    summary: list[str] = Field(default_factory=list)


class BaseConverter(ABC):
    """Abstract base for format converters.

    Subclasses must define:
        name: Short identifier (used with --from flag)
        description: Human-readable description (shown in CLI help)
        detect(): Check if a path matches this format
        convert(): Perform the conversion
    """

    name: str
    description: str

    @abstractmethod
    def detect(self, path: Path) -> bool:
        """Return True if this converter can handle the given path."""

    @abstractmethod
    def convert(self, path: Path) -> ConvertResult:
        """Convert the source at path to HUD format."""
