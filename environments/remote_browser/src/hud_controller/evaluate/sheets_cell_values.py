"""Evaluator to check if specific cells in a Google Sheet have expected values."""

import logging
from typing import Dict, Any, List, Union
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("sheets_cell_values")
async def sheets_cell_values(
    ctx: Context, args: Union[Dict[str, Any], List[Dict[str, Any]]], partial_rewarding: bool = False
):
    """Check if specific cells in a Google Sheet have expected values.

    Args:
        args: Either a dict of cell mappings {"A1": "value", "B2": "value"}
              or a list with a dict [{"A1": "value", "B2": "value"}]
        partial_rewarding: Whether to give partial rewards

    Returns:
        Evaluation result dict with reward, done, and info
    """
    logger.info("Starting sheets_cell_values evaluation")
    logger.info(f"Received args: {args}")

    # Extract cell values from args
    if isinstance(args, list) and len(args) > 0:
        # Handle args as list: args=[{"A1": "value"}]
        cell_values = args[0] if isinstance(args[0], dict) else {}
    elif isinstance(args, dict):
        # Handle args as dict: args={"A1": "value"}
        cell_values = args
    else:
        cell_values = {}

    logger.info(f"Cell values to check: {cell_values}")

    # Get the playwright tool from the environment
    playwright_tool = evaluate.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content="No browser page available",
            info={"error": "No browser page available"},
        )

    page = playwright_tool.page

    # Verify we're on a Google Sheets page
    current_url = page.url
    logger.info(f"Current page URL: {current_url}")

    if "docs.google.com/spreadsheets" not in current_url:
        logger.error(f"Not on a Google Sheets page! URL: {current_url}")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Not on a Google Sheets page! URL: {current_url}",
            info={"error": f"Not on a Google Sheets page! URL: {current_url}"},
        )

    logger.info("Confirmed on Google Sheets page")

    # Validate cell_values
    if not isinstance(cell_values, dict):
        logger.error(
            f"Invalid cell values format: {cell_values}. Expected dictionary of cell references to values."
        )
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Invalid cell values format. Expected dict, got {type(cell_values)}",
            info={"error": f"Invalid cell values format. Expected dict, got {type(cell_values)}"},
        )

    if not cell_values:
        logger.warning("No cell values to check")
        return EvaluationResult(
            reward=1.0,
            done=True,
            content="No cell values to check",
            info={"message": "No cell values to check"},
        )

    # Get all the cell values from the sheet
    try:
        logger.info("Attempting to extract cell values from the page...")

        # JavaScript to get all cell values
        js_code = """
        () => {
            const cells = {};
            const elements = document.querySelectorAll('[id^="waffle-grid-container"] [dir="ltr"]');
            
            elements.forEach(el => {
                const id = el.id;
                if (id && id.includes('-')) {
                    const parts = id.split('-');
                    const cellRef = parts[parts.length - 1];
                    if (cellRef && /^[A-Z]+[0-9]+$/.test(cellRef)) {
                        const spans = el.querySelectorAll('span');
                        let value = '';
                        if (spans.length > 0) {
                            value = Array.from(spans).map(s => s.textContent || '').join('');
                        }
                        cells[cellRef] = value;
                    }
                }
            });
            
            return cells;
        }
        """
        sheet_cells = await page.evaluate(js_code)
        logger.info(f"Found {len(sheet_cells)} cells in the sheet")
        logger.info(f"Sheet cells: {sheet_cells}")

        # Check each expected cell value
        total_cells = len(cell_values)
        matching_cells = 0
        mismatches = []

        for cell_ref, expected_value in cell_values.items():
            actual_value = sheet_cells.get(cell_ref, "")
            logger.info(
                f"Checking {cell_ref}: expected='{expected_value}', actual='{actual_value}'"
            )

            if str(actual_value).strip() == str(expected_value).strip():
                matching_cells += 1
                logger.info(f"✓ {cell_ref} matches")
            else:
                mismatches.append(
                    {
                        "cell": cell_ref,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
                logger.warning(
                    f"✗ {cell_ref} mismatch: expected '{expected_value}', got '{actual_value}'"
                )

        # Calculate reward
        if partial_rewarding:
            reward = matching_cells / total_cells if total_cells > 0 else 0.0
        else:
            reward = 1.0 if matching_cells == total_cells else 0.0

        success = matching_cells == total_cells
        logger.info(
            f"Evaluation result: {matching_cells}/{total_cells} cells match, reward={reward}"
        )

        # Build content message
        if success:
            content = f"All {total_cells} cells match expected values"
        else:
            content = f"{matching_cells}/{total_cells} cells match, {len(mismatches)} mismatches"

        return EvaluationResult(
            reward=float(reward),
            done=success,
            content=content,
            info={
                "success": success,
                "matching_cells": matching_cells,
                "total_cells": total_cells,
                "mismatches": mismatches,
            },
        )

    except Exception as e:
        logger.error(f"Error evaluating sheet cells: {str(e)}", exc_info=True)
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Failed to evaluate: {str(e)}",
            info={"error": f"Failed to evaluate: {str(e)}"},
        )
