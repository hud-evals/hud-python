import os
import logging
from hud.tools.types import EvaluationResult
from hud.server import MCPRouter
from .compare import compare
from ..config import VOLUMES_PATH

logger = logging.getLogger(__name__)
router = MCPRouter()


@router.tool("eval_single")
# async def eval_single(proc_file: str, gt_file: str, answer_position: str):
async def eval_single(id: str, answer_position: str, dataset_path: str = "all_data_912"):
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
        proc_file = os.path.join(
            VOLUMES_PATH, dataset_path, "spreadsheet", id, f"1_{id}_output.xlsx"
        )
        gt_file = os.path.join(VOLUMES_PATH, dataset_path, "spreadsheet", id, f"1_{id}_answer.xlsx")
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
