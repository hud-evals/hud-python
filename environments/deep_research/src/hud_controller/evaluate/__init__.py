"""Evaluation layer for deep research environment."""

from __future__ import annotations

from hud.tools.base import BaseHub

evaluate = BaseHub("evaluate")

from . import url_match, page_contains  # noqa: E402

__all__ = ["evaluate"]

