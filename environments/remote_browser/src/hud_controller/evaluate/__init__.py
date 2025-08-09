"""Evaluation layer for remote browser environment."""

from __future__ import annotations

from hud.tools.base import BaseHub

evaluate = BaseHub("evaluate")

# Import all evaluator functions to register them
from . import (
    url_match,
    page_contains,
    element_exists,
    cookie_exists,
    cookie_match,
    history_length,
    selector_history,
    raw_last_action_is,
    verify_type_action,
    sheet_contains,
    sheets_cell_values,
)

__all__ = ["evaluate"]
