import logging
from hud.tools.types import EvaluationResult
from . import evaluate
from fastmcp import Context
from .compare import compare

logger = logging.getLogger(__name__)


@evaluate.tool("eval_single")
async def eval_single(ctx: Context, proc_file: str, gt_file: str, answer_position: str):
    """
    Evaluate a single SpreadsheetBench instance by comparing output against ground truth.

    Args:
        proc_file: Path to the processed spreadsheet file in the container
        gt_file: Path to the ground truth spreadsheet file in the container
        answer_position: Cell range to compare (e.g., "H3:H5" or "Sheet1!A1:B10")

    Returns:
        EvaluationResult with comparison result
    """
    try:
        result, msg = compare(proc_file, gt_file, answer_position)

        if result:
            content = f"✅ PASS: Cells in range {answer_position} match ground truth"
        else:
            content = f"❌ FAIL: {msg}"

        logger.info(
            f"Evaluation: {proc_file} vs {gt_file} ({answer_position}): {'PASS' if result else 'FAIL'}"
        )

        return EvaluationResult(
            reward=1.0 if result else 0.0,
            done=True,
            isError=False,
            content=content,
            info={
                "proc_file": proc_file,
                "gt_file": gt_file,
                "answer_position": answer_position,
                "passed": result,
                "message": msg if not result else "All cells match",
            },
        )

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return EvaluationResult(
            reward=0.0,
            done=True,
            isError=True,
            content=f"❌ ERROR: {str(e)}",
            info={
                "proc_file": proc_file,
                "gt_file": gt_file,
                "answer_position": answer_position,
                "error": str(e),
            },
        )
