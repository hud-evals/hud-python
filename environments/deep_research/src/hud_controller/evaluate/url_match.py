"""URL match evaluator for deep research environment."""

import logging
from fastmcp import Context
from hud.tools.types import EvaluationResult

from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("url_match")
async def url_match(ctx: Context, expected_substring: str):
    """Reward if current URL contains the expected substring."""
    tool = evaluate.env
    if not tool or not tool.page:
        return EvaluationResult(reward=0.0, done=False, content="No page", info={"success": False})
    try:
        url = tool.page.url
        ok = expected_substring in url
        return EvaluationResult(
            reward=1.0 if ok else 0.0,
            done=ok,
            content=f"URL is {url}",
            info={"success": ok, "url": url},
        )
    except Exception as e:
        logger.error(f"url_match failed: {e}")
        return EvaluationResult(reward=0.0, done=False, content=str(e), info={"success": False})

