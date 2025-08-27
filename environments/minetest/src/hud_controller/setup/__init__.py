"""Setup hub for Minetest environment."""

from hud.tools.base import BaseHub

setup = BaseHub(
    name="setup",
    title="Minetest Setup",
    description="Initialize or configure the Minetest environment",
)

# Import setup tools to register them
from . import game

__all__ = ["setup"]

