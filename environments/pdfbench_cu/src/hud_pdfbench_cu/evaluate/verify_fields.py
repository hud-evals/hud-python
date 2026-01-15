"""PDF field verification evaluator for computer use version.

Note: Since browser PDF editing may not persist to the file,
this evaluator may need to use alternative methods like:
1. Screenshot comparison
2. OCR extraction
3. Browser-based form value extraction

For now, we attempt to read the PDF file if it was saved.
"""

import json
import logging
from typing import Any

import fitz  # PyMuPDF

from hud.tools.types import EvaluationResult
from . import evaluate

logger = logging.getLogger(__name__)


def bbox_key(page_num: int, rect: fitz.Rect) -> str:
    """Create bbox key."""
    return f"{page_num},{round(rect.x0)},{round(rect.y0)},{round(rect.x1)},{round(rect.y1)}"


def normalize_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_checkbox_checked(value: Any) -> bool:
    v = normalize_value(value).lower().strip('/')
    return v not in ['off', 'no', 'false', '0', '']


def values_match(expected: str, actual: Any, fuzzy: bool = True) -> bool:
    expected_norm = normalize_value(expected)
    actual_norm = normalize_value(actual)

    if expected_norm.lower() in ['yes', 'true', '1', 'on']:
        return is_checkbox_checked(actual)
    elif expected_norm.lower() in ['no', 'off', 'false', '0']:
        return not is_checkbox_checked(actual)

    if fuzzy:
        if expected_norm.lower() == actual_norm.lower():
            return True
        if expected_norm.lower() in actual_norm.lower():
            return True

    return expected_norm == actual_norm


def load_solution_json(filepath: str) -> dict[str, str]:
    with open(filepath) as f:
        data = json.load(f)
    return data.get("fields", data)


def get_field_values_by_bbox(filepath: str) -> dict[str, dict[str, Any]]:
    """Extract field values from PDF."""
    doc = fitz.open(filepath)
    field_values = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        for widget in page.widgets():
            key = bbox_key(page_num, widget.rect)
            field_values[key] = {
                "value": widget.field_value,
                "field_name": widget.field_name,
                "field_type": widget.field_type,
            }

    doc.close()
    return field_values


@evaluate.tool("verify_fields")
async def evaluate_verify_fields(
    solution_path: str | None = None,
    pdf_path: str | None = None,
    expected_values: dict[str, str] | None = None,
    fuzzy_match: bool = True,
    partial_credit: bool = True,
) -> EvaluationResult:
    """Evaluate the filled PDF against expected values.

    Args:
        solution_path: Path to solution.json
        pdf_path: Path to the filled PDF (if different from loaded)
        expected_values: Dict mapping bbox keys to expected values
        fuzzy_match: Use case-insensitive substring matching
        partial_credit: Return proportional score
    """
    from ..browser import pdf_browser

    # Determine PDF path
    check_path = pdf_path or pdf_browser.pdf_path
    if not check_path:
        return EvaluationResult(
            reward=0.0,
            done=True,
            content="No PDF path available",
            info={"error": "No PDF loaded"},
        )

    # Load expected values
    sol_path = solution_path or pdf_browser.solution_path
    if expected_values:
        expected = expected_values
    elif sol_path:
        try:
            expected = load_solution_json(sol_path)
        except Exception as e:
            return EvaluationResult(
                reward=0.0,
                done=True,
                content=f"Failed to load solution: {e}",
                info={"error": str(e)},
            )
    else:
        return EvaluationResult(
            reward=0.0,
            done=True,
            content="No solution provided",
            info={"error": "Must provide solution_path or expected_values"},
        )

    # Get actual field values from PDF
    try:
        actual_fields = get_field_values_by_bbox(check_path)
    except Exception as e:
        return EvaluationResult(
            reward=0.0,
            done=True,
            content=f"Failed to read PDF: {e}",
            info={"error": str(e)},
        )

    # Compare fields
    results = {}
    successful_matches = 0
    failed_matches = 0
    missing_fields = 0

    for bbox_key_str, expected_value in expected.items():
        if bbox_key_str in actual_fields:
            actual_value = actual_fields[bbox_key_str]["value"]
            passed = values_match(expected_value, actual_value, fuzzy_match)

            results[bbox_key_str] = {
                "passed": passed,
                "expected": expected_value,
                "actual": actual_value,
                "field_name": actual_fields[bbox_key_str].get("field_name"),
            }

            if passed:
                successful_matches += 1
            else:
                failed_matches += 1
        else:
            missing_fields += 1
            results[bbox_key_str] = {
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
    content = f"Verified {successful_matches}/{total_fields} fields"
    if failed_list:
        content += f" | Failed: {len(failed_list)}"

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
