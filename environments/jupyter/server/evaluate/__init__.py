"""Evaluation tools for the Jupyter environment."""

from hud.tools.base import BaseHub

# Create evaluate hub (tools will be hidden from agents)
evaluate = BaseHub("evaluate")

# Import evaluation tools to register them with the hub
from . import eval_all

__all__ = ["evaluate"]
