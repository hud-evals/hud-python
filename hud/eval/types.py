"""Types and exceptions for the eval module.

Kept separate to avoid circular imports.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# =============================================================================
# Exceptions
# =============================================================================


class ParallelEvalComplete(Exception):
    """Raised by summary context to skip body re-execution after parallel eval.

    This is caught by the eval() context manager to cleanly exit.
    The summary context with results is still accessible after the with block.
    """


# =============================================================================
# Payload Models
# =============================================================================


class EvalPayload(BaseModel):
    """Base payload for eval enter/exit."""

    prompt: str | None = None
    code_snippet: str | None = None
    job_id: str | None = None
    group_id: str | None = None
    variants: dict[str, Any] | None = None
    task_version_id: str | None = None
    metadata: dict[str, Any] | None = None


class EvalExitPayload(EvalPayload):
    """Exit payload with result fields."""

    reward: float | None = None
    success: bool = True
    error_message: str | None = None


class JobEnterPayload(BaseModel):
    """Payload for job/{job_id}/enter - sent once at job start."""

    name: str | None = None
    variants: dict[str, Any] | None = None  # Full variant config
    group: int | None = None


__all__ = [
    "EvalExitPayload",
    "EvalPayload",
    "JobEnterPayload",
    "ParallelEvalComplete",
]
