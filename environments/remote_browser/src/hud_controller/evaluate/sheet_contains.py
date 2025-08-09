"""Evaluator to check if a Google Sheet contains specific text."""

import logging
from typing import Union, List
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("sheet_contains")
async def sheet_contains(
    ctx: Context, args: Union[str, List[str]], partial_rewarding: bool = False
):
    """Check if a Google Sheet contains specific text by copying content to clipboard.

    Args:
        args: Search terms as string or list of strings
        partial_rewarding: Whether to give partial rewards

    Returns:
        Evaluation result
    """
    logger.info("Starting sheet_contains evaluation")

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

    # Process search terms
    search_terms = []
    if isinstance(args, str):
        search_terms = [args]
    elif isinstance(args, list):
        search_terms = args
    else:
        logger.error(f"Invalid args format: {args}. Expected string or list of strings.")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Invalid args format. Expected string or list, got {type(args)}",
            info={"error": f"Invalid args format. Expected string or list, got {type(args)}"},
        )

    if not search_terms:
        logger.error("No search terms provided")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content="No search terms provided",
            info={"error": "No search terms provided"},
        )

    logger.info(f"Search terms to find: {search_terms}")

    try:
        # Select all cells using Ctrl+A
        logger.info("Selecting all cells with Ctrl+A")
        await page.keyboard.press("Control+A")
        await page.wait_for_timeout(500)

        # Copy to clipboard with Ctrl+C
        logger.info("Copying content to clipboard with Ctrl+C")
        await page.keyboard.press("Control+C")
        await page.wait_for_timeout(1000)

        # Get clipboard content
        logger.info("Getting clipboard content")
        clipboard_content = await page.evaluate("() => navigator.clipboard.readText()")

        if not clipboard_content:
            logger.warning("Clipboard content is empty")
            return EvaluationResult(
                reward=0.0,
                done=False,
                content="Clipboard content is empty",
                info={"error": "Clipboard content is empty"},
            )

        logger.info(f"Clipboard content length: {len(clipboard_content)} characters")
        logger.debug(f"First 200 chars: {clipboard_content[:200]}...")

        # Check for search terms
        found_terms = []
        missing_terms = []

        for term in search_terms:
            if term.lower() in clipboard_content.lower():
                found_terms.append(term)
                logger.info(f"✓ Found term: '{term}'")
            else:
                missing_terms.append(term)
                logger.info(f"✗ Missing term: '{term}'")

        # Calculate reward
        if partial_rewarding and len(search_terms) > 0:
            reward = float(len(found_terms)) / len(search_terms)
            logger.info(f"Partial rewarding: {len(found_terms)}/{len(search_terms)} = {reward}")
        elif not missing_terms:
            reward = 1.0
            logger.info("All terms found!")
        else:
            reward = 0.0
            logger.info(f"Missing terms: {missing_terms}")

        success = not missing_terms

        # Build content message
        if success:
            content = "All terms found in sheet"
        else:
            content = f"Missing terms: {missing_terms}"

        return EvaluationResult(
            reward=float(reward),
            done=success,
            content=content,
            info={
                "success": success,
                "found_terms": found_terms,
                "missing_terms": missing_terms,
                "total_terms": len(search_terms),
                "clipboard_length": len(clipboard_content),
            },
        )

    except Exception as e:
        logger.error(f"Exception during sheet_contains evaluation: {str(e)}")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Failed to evaluate: {str(e)}",
            info={"error": f"Failed to evaluate: {str(e)}"},
        )
