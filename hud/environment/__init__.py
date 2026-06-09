"""HUD environment authoring runtime."""

from hud.capabilities import Capability
from hud.server import MCPRouter

from .env import Environment
from .workspace import DEFAULT_SYSTEM_MOUNTS, Mount, MountKind, Workspace

ToolRouter = MCPRouter

__all__ = [
    "DEFAULT_SYSTEM_MOUNTS",
    "Capability",
    "Environment",
    "MCPRouter",
    "Mount",
    "MountKind",
    "ToolRouter",
    "Workspace",
]
