"""Schema models for rollout collection."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "hud.rollout.v1"


def make_rollout_id(source: str, task_index: int, repeat_index: int, prompt: str) -> str:
    """Build a stable rollout identifier from task identity."""
    seed = f"{source}|{task_index}|{repeat_index}|{prompt}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"rollout_{digest[:16]}"


class RolloutRecord(BaseModel):
    """Serialized rollout record for offline RL/RFT pipelines."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    rollout_id: str
    source: str
    task_index: int
    repeat_index: int
    task_id: str | None = None
    prompt: str
    reward: float = 0.0
    done: bool = True
    is_error: bool = False
    content: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    messages: list[Any] = Field(default_factory=list)
