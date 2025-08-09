"""Evaluator to verify a click-then-type action sequence."""

import logging
from typing import Optional
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("verify_type_action")
async def verify_type_action(
    ctx: Context,
    expected_text: str,
    selector: Optional[str] = None,
    partial_rewarding: bool = False,
):
    """Check for a sequence: first click on element, then type text into it.

    Args:
        expected_text: The expected text that should have been typed
        selector: Optional selector to check (if not provided, checks last type action)
        partial_rewarding: Whether to give partial rewards

    Returns:
        Standard evaluation result with reward between 0.0 and 1.0
    """
    logger.info("Starting verify_type_action evaluation")

    expected_value = expected_text

    if not expected_value:
        logger.error("No expected text provided")
        return EvaluationResult(
            reward=0.0, done=False, content="No expected text provided", info={"success": False}
        )

    logger.info(f"Looking for type action with text: {expected_value}")
    if selector:
        logger.info(f"Checking for specific selector: {selector}")

    # Get the playwright tool from the environment
    playwright_tool = evaluate.env
    if (
        not playwright_tool
        or not hasattr(playwright_tool, "action_history")
        or not playwright_tool.action_history
    ):
        logger.error("No playwright tool available")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content="No playwright tool available",
            info={"error": "No playwright tool available"},
        )

    action_history = playwright_tool.action_history

    logger.info(f"Total actions in history: {len(action_history)}")

    if len(action_history) == 0:
        logger.info("No actions in history")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content="No actions in history",
            info={"success": False, "action_count": 0},
        )

    # Look for the most recent type action
    for i in range(len(action_history) - 1, -1, -1):
        action = action_history[i]

        if action.get("type") == "type":
            action_details = action.get("details", {})
            typed_text = action_details.get("text", "")
            action_selector = action_details.get("selector", "")

            # Check if selector matches (if specified)
            if selector and action_selector != selector:
                continue

            # Check if typed text matches
            if str(typed_text) == str(expected_value):
                logger.info(f"✓ Found matching type action at index {i}")
                logger.info(f"  Selector: {action_selector}")
                logger.info(f"  Text: '{typed_text}'")

                return EvaluationResult(
                    reward=1.0,
                    done=True,
                    content="Found matching type action",
                    info={
                        "success": True,
                        "typed_text": typed_text,
                        "selector": action_selector,
                        "action_index": i,
                    },
                )
            elif not selector:
                # If no specific selector required, any mismatch is a failure
                logger.info(f"✗ Found type action but text mismatch")
                logger.info(f"  Expected: '{expected_value}'")
                logger.info(f"  Got: '{typed_text}'")

                if partial_rewarding:
                    return EvaluationResult(
                        reward=0.5,
                        done=False,
                        content="Text mismatch",
                        info={
                            "success": False,
                            "expected": expected_value,
                            "actual": typed_text,
                            "selector": action_selector,
                        },
                    )

    logger.info("No matching type action found")
    return EvaluationResult(
        reward=0.0,
        done=False,
        content="No matching type action found",
        info={"success": False, "expected_text": expected_value, "required_selector": selector},
    )
