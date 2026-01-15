"""PDF field verification evaluator."""

import json
import logging
from typing import Any

from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


def normalize_value(value: Any) -> str:
    """Normalize a value to string for comparison."""
    if value is None:
        return ""
    return str(value).strip()


def is_checkbox_checked(value: Any) -> bool:
    """Determine if a checkbox value represents 'checked' state."""
    v = normalize_value(value).lower().strip('/')
    return v not in ['off', 'no', 'false', '0', '']


def values_match(expected: str, actual: Any, fuzzy: bool = True) -> bool:
    """Compare expected and actual values."""
    expected_norm = normalize_value(expected)
    actual_norm = normalize_value(actual)

    # Handle checkbox values
    if expected_norm.lower() in ['yes', 'true', '1', 'on']:
        return is_checkbox_checked(actual)
    elif expected_norm.lower() in ['no', 'off', 'false', '0']:
        return not is_checkbox_checked(actual)

    # Text field comparison
    if fuzzy:
        if expected_norm.lower() == actual_norm.lower():
            return True
        if expected_norm.lower() in actual_norm.lower():
            return True

    return expected_norm == actual_norm


def load_solution_json(filepath: str) -> dict[str, str]:
    """Load solution JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get("fields", data)


@evaluate.tool("verify_fields")
async def evaluate_verify_fields(
    solution_path: str | None = None,
    expected_values: dict[str, str] | None = None,
    fuzzy_match: bool = True,
    partial_credit: bool = True,
    strict_empty: bool = False,
) -> EvaluationResult:
    """Evaluate the filled PDF against expected values.

    Args:
        solution_path: Path to solution.json
        expected_values: Dict mapping bbox keys to expected values
        fuzzy_match: Use case-insensitive substring matching
        partial_credit: Return proportional score vs all-or-nothing
        strict_empty: Verify unlisted fields are empty/unchanged
    """
    ctx = evaluate.env

    if not ctx.doc:
        return EvaluationResult(
            reward=0.0,
            done=True,
            content="No PDF loaded",
            info={"error": "No PDF loaded. Call setup first."},
        )

    # Save current state for verification
    temp_path = "/tmp/verify_temp.pdf"
    ctx.doc.save(temp_path)

    # Load expected values
    if expected_values:
        expected = expected_values
    elif solution_path:
        expected = load_solution_json(solution_path)
    elif ctx.solution_path:
        expected = load_solution_json(ctx.solution_path)
    else:
        return EvaluationResult(
            reward=0.0,
            done=True,
            content="No solution provided",
            info={"error": "Must provide solution_path or expected_values"},
        )

    # Get actual field values
    actual_fields = ctx.get_field_values_by_bbox(temp_path)

    # Get original values if needed
    original_fields = {}
    if strict_empty and ctx.original_pdf_path:
        original_fields = ctx.get_field_values_by_bbox(ctx.original_pdf_path)

    # Track results
    results = {}
    successful_matches = 0
    failed_matches = 0
    missing_fields = 0

    for bbox_key, expected_value in expected.items():
        if bbox_key in actual_fields:
            actual_value = actual_fields[bbox_key]["value"]
            passed = values_match(expected_value, actual_value, fuzzy_match)

            results[bbox_key] = {
                "passed": passed,
                "expected": expected_value,
                "actual": actual_value,
                "field_name": actual_fields[bbox_key].get("field_name"),
            }

            if passed:
                successful_matches += 1
            else:
                failed_matches += 1
        else:
            missing_fields += 1
            results[bbox_key] = {
                "passed": False,
                "expected": expected_value,
                "actual": None,
                "error": "Field not found",
            }

    # Calculate reward
    total_fields = len(expected)
    if partial_credit:
        reward = successful_matches / total_fields if total_fields > 0 else 0.0
    else:
        reward = 1.0 if successful_matches == total_fields else 0.0

    # Build content message
    failed_list = [k for k, v in results.items() if not v.get("passed", False)]
    content = f"Verified {successful_matches}/{total_fields} fields correctly"
    if failed_list:
        content += f" | Failed: {', '.join(failed_list[:3])}"
        if len(failed_list) > 3:
            content += f"... (+{len(failed_list) - 3} more)"

    return EvaluationResult(
        reward=reward,
        done=True,
        content=content,
        info={
            "total_fields": total_fields,
            "successful_matches": successful_matches,
            "failed_matches": failed_matches,
            "missing_fields": missing_fields,
            "results": results,
        },
    )
