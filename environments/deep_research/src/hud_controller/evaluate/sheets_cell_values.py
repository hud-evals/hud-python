"""Evaluator to check if specific cells in a Google Sheet have expected values."""

import asyncio
import logging
from typing import Dict, Any, List, Union
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("sheets_cell_values")
async def sheets_cell_values(
    ctx: Context, args: Union[Dict[str, Any], List[Dict[str, Any]]], partial_rewarding: bool = True
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
    context = page.context

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

    # Try to navigate to the ANSWER sheet tab with retries
    logger.info("=== ANSWER Sheet Navigation ===")
    max_attempts = 3
    answer_navigation_successful = False

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(
                f"Attempt {attempt}/{max_attempts}: Attempting to find and navigate to ANSWER sheet tab..."
            )

            # Look for the ANSWER sheet tab using the selector
            answer_tab_selector = 'span.docs-sheet-tab-name:has-text("ANSWER")'
            logger.info(f"Searching for ANSWER tab with selector: {answer_tab_selector}")

            # Check if the ANSWER tab exists
            answer_tab_exists = await page.locator(answer_tab_selector).count() > 0
            logger.info(
                f"ANSWER tab search result (attempt {attempt}): {'Found' if answer_tab_exists else 'Not found'}"
            )

            if answer_tab_exists:
                logger.info(f"✅ Found ANSWER sheet tab on attempt {attempt}, clicking on it...")
                await page.locator(answer_tab_selector).click()
                logger.info("Clicked on ANSWER tab, waiting for sheet to switch...")

                # Wait a bit for the sheet to switch
                try:
                    await page.wait_for_timeout(1000)
                except Exception as timeout_error:
                    logger.debug(f"Timeout error (continuing): {timeout_error}")
                    await asyncio.sleep(1)
                logger.info(f"✅ Successfully navigated to ANSWER sheet on attempt {attempt}")
                answer_navigation_successful = True
                break
            else:
                logger.warning(f"⚠️ ANSWER sheet tab not found on attempt {attempt}")

                if attempt < max_attempts:
                    logger.info(f"Waiting 500ms before retry {attempt + 1}...")
                    try:
                        await page.wait_for_timeout(500)
                    except Exception as timeout_error:
                        logger.debug(f"Timeout error (continuing): {timeout_error}")
                        await asyncio.sleep(0.5)

        except Exception as nav_error:
            logger.error(
                f"❌ Error navigating to ANSWER sheet on attempt {attempt}: {str(nav_error)}"
            )

            if attempt < max_attempts:
                logger.info(f"Waiting 2500ms before retry {attempt + 1}...")
                try:
                    await page.wait_for_timeout(2500)
                except Exception as timeout_error:
                    logger.debug(f"Timeout error (continuing): {timeout_error}")
                    await asyncio.sleep(2.5)

    if not answer_navigation_successful:
        logger.warning(
            f"⚠️ Failed to navigate to ANSWER sheet after {max_attempts} attempts, proceeding with current sheet"
        )

    # Wait for sheet to fully load
    logger.info("Waiting for sheet to fully load...")
    try:
        # Wait for grid container to be present
        await page.wait_for_selector(".grid-container", timeout=20000)
        logger.info("Sheet grid container loaded")

        # Additional wait for cells to populate
        try:
            await page.wait_for_timeout(2000)
        except Exception as timeout_error:
            logger.debug(f"Timeout error (continuing): {timeout_error}")
            await asyncio.sleep(2)
    except Exception as e:
        logger.warning(f"Timeout waiting for sheet to load: {str(e)}")
        # Still proceed, but with a longer fallback wait
        await asyncio.sleep(5)

    # Extract sheet content using clipboard method
    try:
        logger.info("=== File Content Extraction ===")

        # Grant clipboard permissions
        try:
            await context.grant_permissions(["clipboard-read", "clipboard-write"])
            logger.info("Granted clipboard read-write permissions")
        except Exception as perm_error:
            logger.warning(f"Failed to grant permissions: {str(perm_error)}")

        logger.info("Extracting page contents")

        # Clear any selection and focus on the sheet
        await page.keyboard.press("Escape")

        # Click on the sheet body to ensure focus
        await page.locator("body").click(force=True)

        # Click on the sheet container
        await page.click(".fixed4-inner-container")

        logger.info("Selecting all content with Ctrl+A")

        # Select all content
        await page.keyboard.press("Control+A")

        # Wait for 1 second
        await asyncio.sleep(1)

        # Copy to clipboard
        await page.keyboard.press("Control+C")

        # Wait for 1 second
        await asyncio.sleep(1)

        # Get clipboard content
        clipboard_content = await page.evaluate("() => navigator.clipboard.readText()")
        logger.info(f"Successfully extracted {len(clipboard_content)} characters from file")

        # Parse the clipboard content to extract cell values
        # Split content into rows (by newlines)
        rows = clipboard_content.rstrip("\n").split("\n")
        logger.info(f"Split file content into {len(rows)} rows")

        # Show first few rows for debugging
        if len(rows) > 0:
            logger.info("First few rows of content:")
            for i, row in enumerate(rows[:3]):
                row_preview = row.replace("\t", " | ")[:100]
                logger.info(f"  Row {i + 1}: '{row_preview}{'...' if len(row) > 100 else ''}'")
            if len(rows) > 3:
                logger.info(f"  ... and {len(rows) - 3} more rows")

        logger.info("=== Cell Reference Parsing ===")

        # Parse cell references to get row and column indices
        actual_values = {}
        for cell_ref, expected_value in cell_values.items():
            logger.info(f"Processing cell reference: '{cell_ref}' -> expected: '{expected_value}'")

            # Extract row and column from cell reference (e.g., "A1" -> row=0, col=0)
            if len(cell_ref) < 2 or not cell_ref[0].isalpha() or not cell_ref[1:].isdigit():
                logger.error(
                    f"❌ Invalid cell reference format: '{cell_ref}' (expected format: A1, B2, etc.)"
                )
                actual_values[cell_ref] = None
                continue

            col_letter = cell_ref[0].upper()
            row_num = int(cell_ref[1:]) - 1  # Convert to 0-indexed
            col_num = ord(col_letter) - ord("A")  # Convert A->0, B->1, etc.

            logger.info(
                f"  Parsed '{cell_ref}' -> row={row_num + 1} (0-indexed: {row_num}), col={col_letter} (0-indexed: {col_num})"
            )

            # Check if the row exists in our parsed content
            if row_num < len(rows):
                logger.info(f"  Row {row_num + 1} exists in content")
                # Split the row into cells (by tabs)
                cells = rows[row_num].split("\t")
                logger.info(f"  Row {row_num + 1} has {len(cells)} columns")

                # Check if the column exists in this row
                if col_num < len(cells):
                    actual_values[cell_ref] = cells[col_num]
                    logger.info(f"  ✅ Found value for {cell_ref}: '{actual_values[cell_ref]}'")
                else:
                    logger.warning(
                        f"  ❌ Column {col_letter} (index {col_num}) not found in row {row_num + 1} (has {len(cells)} columns)"
                    )
                    actual_values[cell_ref] = ""
            else:
                logger.warning(
                    f"  ❌ Row {row_num + 1} not found in content (has {len(rows)} rows)"
                )
                actual_values[cell_ref] = ""

        logger.info("=== Cell Value Comparison ===")

        # Check each expected cell value
        total_cells = len(cell_values)
        matching_cells = 0
        mismatches = []

        for cell_ref, expected_value in cell_values.items():
            actual_value = actual_values.get(cell_ref, "")
            logger.info(f"Comparing cell {cell_ref}:")
            logger.info(f"  Expected: '{expected_value}' (type: {type(expected_value)})")
            logger.info(f"  Actual:   '{actual_value}' (type: {type(actual_value)})")

            if actual_value is None:
                mismatch_msg = f"Cell {cell_ref} not found"
                mismatches.append({"cell": cell_ref, "expected": expected_value, "actual": ""})
                logger.info(f"  ❌ {mismatch_msg}")
            elif str(actual_value).strip() == str(expected_value).strip():
                matching_cells += 1
                logger.info(
                    f"  ✅ MATCH: '{str(actual_value).strip()}' == '{str(expected_value).strip()}'"
                )
            else:
                mismatches.append(
                    {
                        "cell": cell_ref,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
                logger.info(
                    f"  ❌ VALUE MISMATCH: '{str(actual_value).strip()}' != '{str(expected_value).strip()}'"
                )

        # Calculate reward
        if partial_rewarding and total_cells > 0:
            reward = matching_cells / total_cells
            logger.info(f"✅ Partial rewarding: {matching_cells}/{total_cells} = {reward}")
        elif matching_cells == total_cells:
            reward = 1.0
            logger.info("✅ ALL cells match expected values!")
        else:
            reward = 0.0
            logger.info("❌ NOT all cells match expected values")
            logger.info(f"Mismatches: {mismatches}")

        success = matching_cells == total_cells
        logger.info(f"Final reward: {reward}")

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
