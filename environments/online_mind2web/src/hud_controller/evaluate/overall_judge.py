import logging
from fastmcp import Context
from hud.tools.types import EvaluationResult
from . import evaluate
from .autonomous_eval import autonomous_eval
from .webjudge import webjudge_eval

logger = logging.getLogger(__name__)


@evaluate.tool("overall_judge")
async def overall_judge(ctx: Context, task_description: dict | str) -> dict | EvaluationResult:
    """Judge and return the results from all evalution methods

    Args:
        ctx: Context, passed automatically
        task_description: Task description (dict or JSON string)

    Returns:
        Dict containing rewards and info
    """
    evaluation_methods = [autonomous_eval, webjudge_eval]

    info = {}
    reward = 0.0
    errors = 0
    done = 0.0
    n = float(len(evaluation_methods))

    try:
        for f in evaluation_methods:
            r: EvaluationResult = await f(ctx, task_description)
            reward += r.reward
            errors += r.isError
            done += int(r.done)
            info[f.__name__] = {
                "reward": r.reward,
                "done": r.done,
                "isError": r.isError,
                "info": r.info,
            }

        return EvaluationResult(
            reward=reward / n, done=(done >= n / 2), info=info, isError=(errors > 0)
        )
    except Exception as e:
        logger.error(f"Overall evaluation failed: {e}")
        return EvaluationResult(isError=True, info={"Exception": str(e)})
