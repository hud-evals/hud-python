"""Selector history evaluator for remote browser environment."""

import logging
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("selector_history")
async def selector_history(ctx: Context, index: int, expected_selector: str):
    """Check if selector at index matches expected.

    Args:
        index: Index in selector history (0-based)
        expected_selector: Expected selector string

    Returns:
        Standard evaluation result with reward between 0.0 and 1.0
    """
    logger.info(f"Evaluating selector_history: index={index}, expected={expected_selector}")

    # Get the playwright tool from the environment
    playwright_tool = evaluate.env
    if not playwright_tool:
        logger.error("No playwright tool available")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content="No playwright tool available",
            info={"error": "No playwright tool available"},
        )

    # Get selector history
    selector_history = (
        playwright_tool.selector_history if hasattr(playwright_tool, "selector_history") else []
    )

    # Check if index is valid
    if index < 0 or index >= len(selector_history):
        logger.info(f"No selector found at index {index}")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"No selector found at index {index}",
            info={
                "success": False,
                "expected_selector": expected_selector,
                "selector_history_length": len(selector_history),
            },
        )

    # Get selector at index
    actual_selector = selector_history[index]

    # Check if selector matches
    success = actual_selector == expected_selector

    if success:
        content = f"Selector at index {index} matches: {expected_selector}"
    else:
        content = f"Selector at index {index} '{actual_selector}' does not match expected '{expected_selector}'"

    logger.info(f"Selector history evaluation: {content}")

    return EvaluationResult(
        reward=1.0 if success else 0.0,
        done=success,
        content=content,
        info={
            "success": success,
            "actual_selector": actual_selector,
            "expected_selector": expected_selector,
            "index": index,
            "selector_history_length": len(selector_history),
        },
    )
