"""Agent that runs the ``claude`` CLI over SSH."""

from hud.agents.types import ClaudeCLIConfig

from .agent import ClaudeCLIAgent

__all__ = ["ClaudeCLIAgent", "ClaudeCLIConfig"]
