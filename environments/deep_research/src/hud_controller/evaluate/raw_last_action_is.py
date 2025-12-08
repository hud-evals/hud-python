"""Raw last action evaluator for remote browser environment."""

import logging
from typing import Optional, Dict, Any
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("raw_last_action_is")
async def raw_last_action_is(
    ctx: Context, expected_action: str, expected_details: Optional[Dict[str, Any]] = None
):
    """Check if the last action matches expected.

    Args:
        expected_action: Expected action type (e.g., "click", "type", "navigate")
        expected_details: Optional expected details of the action

    Returns:
        Standard evaluation result with reward between 0.0 and 1.0
    """
    logger.info(f"Evaluating raw_last_action_is: expected={expected_action}")

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

    # Get action history
    action_history = (
        playwright_tool.action_history if hasattr(playwright_tool, "action_history") else []
    )

    if not action_history:
        logger.info("No actions have been performed yet")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content="No actions have been performed",
            info={"success": False, "expected_action": expected_action},
        )

    # Get last action
    last_action = action_history[-1]

    # Check if action type matches
    action_matches = last_action["type"] == expected_action
    details_match = True

    # Check details if provided
    if expected_details and action_matches:
        actual_details = last_action.get("details", {})
        for key, expected_value in expected_details.items():
            if actual_details.get(key) != expected_value:
                details_match = False
                break

    success = action_matches and details_match

    if success:
        content = f"Last action matches: {expected_action}"
    else:
        if not action_matches:
            content = (
                f"Last action '{last_action['type']}' does not match expected '{expected_action}'"
            )
        else:
            content = f"Action matches but details do not match"

    logger.info(f"Last action evaluation: {content}")

    return EvaluationResult(
        reward=1.0 if success else 0.0,
        done=success,
        content=content,
        info={
            "success": success,
            "last_action": last_action,
            "expected_action": expected_action,
            "expected_details": expected_details,
        },
    )
