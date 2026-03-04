"""Types for chat service orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hud.eval.task import Task


class SessionState(str, Enum):
    """Lifecycle states for managed chat sessions."""

    ACTIVE = "active"
    FINISHING = "finishing"
    FINISHED = "finished"
    EXPIRED = "expired"


@dataclass(slots=True)
class ChatDefinition:
    """Configuration for building Chat instances.

    A chat definition is the primitive tuple of:
    - task template (scenario + environment + defaults)
    - model
    - optional agent parameters and display metadata
    """

    name: str
    task: Task
    model: str
    agent_params: dict[str, Any] = field(default_factory=dict)
    display_name: str | None = None
    description: str | None = None
    tool_properties: dict[str, Any] | None = None
    tool_required: list[str] | None = None

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("ChatDefinition.name must be non-empty")
        if not self.model.strip():
            raise ValueError("ChatDefinition.model must be non-empty")
