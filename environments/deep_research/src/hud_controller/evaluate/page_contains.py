"""Page contains evaluator for deep research environment."""

import logging
from typing import List, Union

from fastmcp import Context
from hud.tools.types import EvaluationResult

from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("page_contains")
async def page_contains(ctx: Context, search_terms: Union[str, List[str]], partial_rewarding: bool = True):
    """Check if the page contains specific text."""
    tool = evaluate.env
    if not tool or not tool.page:
        return EvaluationResult(reward=0.0, done=False, content="No page", info={"success": False})
    try:
        content = await tool.page.content()
        terms = [search_terms] if isinstance(search_terms, str) else list(search_terms)
        found = [t for t in terms if t in content]
        not_found = [t for t in terms if t not in content]
        if partial_rewarding and terms:
            reward = len(found) / len(terms)
        else:
            reward = 1.0 if not not_found else 0.0
        msg = (
            "All terms found" if reward == 1.0 else (f"Found {len(found)} of {len(terms)} terms" if reward > 0 else "No terms found")
        )
        return EvaluationResult(
            reward=float(reward),
            done=reward == 1.0,
            content=msg,
            info={
                "success": reward > 0,
                "found_terms": found,
                "not_found_terms": not_found,
                "total_terms": len(terms),
            },
        )
    except Exception as e:
        logger.error(f"page_contains failed: {e}")
        return EvaluationResult(reward=0.0, done=False, content=str(e), info={"success": False})

