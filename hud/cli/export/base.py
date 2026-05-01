"""Abstract base classes for format exporters.

Mirrors the structure of hud/cli/convert/base.py but goes the inverse
direction: HUD environments + tasksets → external benchmark format.

BaseExporter.export() is pure: it takes a structured ExportInput and
returns an ExportResult containing files-to-write plus a manifest.
The CLI shell writes the result to disk; the platform pipeline streams
it to S3/Supabase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hud.eval.task import Task

__all__ = ["BaseExporter", "ExportInput", "ExportResult"]


class ExportInput(BaseModel):
    """Structured input for an export run.

    Constructed by the CLI (from a local repo) or by the platform
    pipeline (from taskset API + image registry + trace store).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tasks: list[Task]
    env_image: str
    env_platform: str | None = None
    env_runtime_command: list[str] | None = None
    env_required_env: list[str] = Field(default_factory=list)
    repo_root: Path | None = None
    rendered_prompts: dict[str, str] = Field(default_factory=dict)
    taskset_name: str
    taskset_id: str | None = None
    task_subset: list[str] | None = None
    bundle_name: str | None = None


class ExportResult(BaseModel):
    """Files-to-write plus a manifest.

    `files` keys are POSIX-style relative paths under the export root.
    Values may be str (text) or bytes (binary). The CLI's write_result()
    materializes these to disk; the platform streams them into a tarball.
    """

    files: dict[str, str | bytes]
    manifest: dict[str, Any]
    sample_run_script: str
    summary: list[str] = Field(default_factory=list)


class BaseExporter(ABC):
    """Abstract base for HUD-to-external format exporters."""

    name: str
    description: str

    @abstractmethod
    def export(self, inp: ExportInput) -> ExportResult:
        """Produce an ExportResult from the given input. Must be pure."""
