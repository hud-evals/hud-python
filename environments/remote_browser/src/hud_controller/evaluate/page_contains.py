"""Page contains evaluator for remote browser environment."""

import logging
from typing import Union, List
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("page_contains")
async def page_contains(
    ctx: Context, search_terms: Union[str, List[str]], partial_rewarding: bool = False
):
    """Check if the page contains specific text.

    Args:
        search_terms: Text to search for (string or list of strings)
        partial_rewarding: If True, give partial credit for finding some terms

    Returns:
        Evaluation result with reward between 0.0 and 1.0
    """
    logger.info(f"Evaluating page_contains for terms: {search_terms}")

    # Get the playwright tool from the environment
    playwright_tool = evaluate.env
    if not playwright_tool or not hasattr(playwright_tool, "page") or not playwright_tool.page:
        logger.error("No browser page available")
        return EvaluationResult(
            reward=0.0, done=False, content="No browser page available", info={"success": False}
        )

    # Get page content
    try:
        content = await playwright_tool.page.content()
        logger.info(f"Page content retrieved, length: {len(content)}")
    except Exception as e:
        logger.error(f"Failed to get page content: {e}")
        return EvaluationResult(
            reward=0.0,
            done=False,
            content=f"Failed to get page content: {str(e)}",
            info={"success": False},
        )

    # Normalize search terms to list
    if isinstance(search_terms, str):
        terms = [search_terms]
    else:
        terms = search_terms

    # Search for terms
    found_terms = []
    not_found_terms = []

    for term in terms:
        if term in content:
            found_terms.append(term)
            logger.info(f"✅ Found term: '{term}'")
        else:
            not_found_terms.append(term)
            logger.info(f"❌ Term not found: '{term}'")

    # Calculate reward
    if partial_rewarding and terms:
        reward = len(found_terms) / len(terms)
    else:
        reward = 1.0 if len(not_found_terms) == 0 else 0.0

    # Build content message
    if reward == 1.0:
        content_msg = "All terms found on page"
    elif reward > 0:
        content_msg = f"Found {len(found_terms)} of {len(terms)} terms"
    else:
        content_msg = "No terms found on page"

    logger.info(f"Page contains evaluation complete. Reward: {reward}")

    return EvaluationResult(
        reward=float(reward),
        done=reward == 1.0,
        content=content_msg,
        info={
            "success": reward > 0,
            "found_terms": found_terms,
            "not_found_terms": not_found_terms,
            "total_terms": len(terms),
            "partial_rewarding": partial_rewarding,
        },
    )
