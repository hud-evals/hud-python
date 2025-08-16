"""Setup module for browser environment."""

from hud.tools import SetupTool

# Create global setup tool instance
setup_tool = SetupTool(
    name="setup",
    title="Browser Setup",
    description="Initialize or configure the browser environment",
)

# Convenience decorator
setup = setup_tool.register

# Import all setup modules to register their functions
from . import todo, apps

__all__ = ["setup_tool", "setup"]
