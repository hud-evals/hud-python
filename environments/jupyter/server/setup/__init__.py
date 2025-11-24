"""Setup tools for the sheet environment."""

from hud.tools.base import BaseHub

# Create setup hub (tools will be hidden from agents)
setup = BaseHub("setup")

# No Setup Tools for Jupyter Environment for now

__all__ = ["setup"]
