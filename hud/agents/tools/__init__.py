"""Shared primitives for agent-owned harness tools."""

from __future__ import annotations

from .base import (
    AgentTool,
    AgentTools,
    AgentToolSpec,
)
from .capabilities import (
    CapabilityEntry,
    EnvironmentCapability,
    GroupedCapabilityMixin,
    ToolMetadata,
    discover_environment_capabilities,
)
from .hosted import HostedTool

__all__ = [
    "AgentTool",
    "AgentToolSpec",
    "AgentTools",
    "CapabilityEntry",
    "EnvironmentCapability",
    "GroupedCapabilityMixin",
    "HostedTool",
    "ToolMetadata",
    "discover_environment_capabilities",
]
