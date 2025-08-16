"""Evaluators module for browser environment."""

from hud.tools import EvaluateTool

# Create global evaluate tool instance
evaluate_tool = EvaluateTool(
    name="evaluate", title="Browser Evaluator", description="Evaluate the current browser state"
)

# Convenience decorator
evaluator = evaluate_tool.register

# Import all evaluator modules to register their functions
from . import todo

__all__ = ["evaluate_tool", "evaluator"]
