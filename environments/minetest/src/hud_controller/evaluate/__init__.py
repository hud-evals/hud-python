"""Evaluate hub for Minetest environment."""

from hud.tools.base import BaseHub

evaluate = BaseHub(
    name="evaluate",
    title="Minetest Evaluation",
    description="Evaluate readiness and running state of the Minetest environment",
)

from . import health  # Register evaluators

__all__ = ["evaluate"]

