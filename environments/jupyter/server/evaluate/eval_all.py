import os
import logging
from pathlib import Path
from .compare import compare
from .generalize import generalize_code
from ..config import VOLUMES_PATH, SOLUTIONS_PATH
from ..tools import JupyterToolWithRecord
from . import evaluate

logger = logging.getLogger(__name__)


@evaluate.tool("eval_all")
async def eval_all(id: str, answer_position: str, dataset_path: str = "all_data_912"):
    """
    Evaluate solution on all three instances (generalization test).

    Similar to SpreadsheetBench's run_solution():
    1. Generalize code from instance 1 to instances 2 & 3
    2. Execute all three solutions
    3. Evaluate each output against ground truth

    Args:
        id: Task ID
        dataset_path: Path to dataset directory

    Returns:
        EvaluationResult with aggregated results for all instances
    """
    try:
        # Connect to the shared kernel
        jupyter_tool = JupyterToolWithRecord.from_shared_kernel("SpreadSheetBench")

        dataset_dir = Path(VOLUMES_PATH) / dataset_path
        spreadsheet_dir = dataset_dir / "spreadsheet" / id
        if not spreadsheet_dir.exists():
            raise FileNotFoundError(f"Spreadsheet directory not found: {spreadsheet_dir}")

        # Step 1: Generalize code
        logger.info(f"Generalizing solution for task {id}")
        gen_results = generalize_code(id)
        if "error" in gen_results:
            raise RuntimeError(f"Generalization failed: {gen_results['error']}")

        # Step 2: Execute and evaluate all three instances
        results = {}
        total_passed = 0
        for i in range(1, 4):
            instance_key = f"instance_{i}"
            solution_path = os.path.join(SOLUTIONS_PATH, f"{i}_solution.py")
            output_file = spreadsheet_dir / f"{i}_{id}_output.xlsx"
            answer_file = spreadsheet_dir / f"{i}_{id}_answer.xlsx"

            # Execute solution
            logger.info(f"Executing solution for instance {i}")
            try:
                with open(solution_path, "r") as f:
                    solution_code = f.read()
                exec_result = await jupyter_tool._execute(solution_code)

                # Check for execution errors
                is_error = (
                    "-----" in exec_result or "Error" in exec_result or "Traceback" in exec_result
                )
                if is_error:
                    results[instance_key] = {
                        "passed": False,
                        "execution_error": exec_result[:500],  # Truncate long errors
                        "reward": 0.0,
                    }
                    continue

                # Evaluate output
                if not output_file.exists():
                    results[instance_key] = {
                        "passed": False,
                        "error": "Output file not created by solution",
                        "reward": 0.0,
                    }
                    continue

                # Compare with ground truth (get answer_position from first instance)
                passed, msg = compare(str(output_file), str(answer_file), answer_position)

                results[instance_key] = {
                    "passed": passed,
                    "message": msg,
                    "reward": 1.0 if passed else 0.0,
                }

                if passed:
                    total_passed += 1

            except Exception as e:
                results[instance_key] = {"passed": False, "error": str(e), "reward": 0.0}

        # Calculate final score
        total_instances = 3
        success_rate = total_passed / total_instances

        # Build summary
        summary = f"✅ Passed: {total_passed}/{total_instances} instances\n"
        summary += f"📊 Success Rate: {success_rate:.1%}\n\n"

        for i in range(1, 4):
            instance_key = f"instance_{i}"
            if instance_key in results:
                result = results[instance_key]
                status = "✅ PASS" if result.get("passed", False) else "❌ FAIL"
                summary += f"Instance {i}: {status}\n"
                if not result.get("passed", False):
                    error_msg = (
                        result.get("error")
                        or result.get("execution_error")
                        or result.get("message", "Unknown error")
                    )
                    summary += f"  Error: {error_msg[:200]}\n"

        logger.info(f"Evaluation complete: {total_passed}/{total_instances} passed")

        # Return plain dict (like browser environment) instead of EvaluationResult
        return {
            "reward": success_rate,
            "done": True,
            "isError": False,
            "content": summary,
            "info": {
                "task_id": id,
                "total_passed": total_passed,
                "total_instances": total_instances,
                "success_rate": success_rate,
                "generalization": gen_results,
                "instance_results": results,
            },
        }

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        # Return plain dict (like browser environment) instead of EvaluationResult
        return {
            "reward": 0.0,
            "done": True,
            "isError": True,
            "content": f"❌ ERROR: {str(e)}",
            "info": {"task_id": id, "error": str(e)},
        }
