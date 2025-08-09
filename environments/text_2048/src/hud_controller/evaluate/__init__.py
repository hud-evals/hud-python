"""Evaluation layer for 2048 environment."""

from __future__ import annotations

from hud.tools.base import BaseHub

evaluate = BaseHub("evaluate")

# Import all evaluator functions to register them
from . import efficiency, max_number

__all__ = ["evaluate"]
