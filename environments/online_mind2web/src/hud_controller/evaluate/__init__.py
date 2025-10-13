"""Evaluation layer for remote browser environment."""

from __future__ import annotations

from hud.tools.base import BaseHub

evaluate = BaseHub("evaluate")

from . import autonomous_eval, webjudge, overall_judge

__all__ = ["evaluate"]
