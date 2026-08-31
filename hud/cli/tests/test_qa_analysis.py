"""Parse standard QA-agent result blobs into review chrome."""

from __future__ import annotations

from hud.cli.qa_analysis import is_standard_result_blob, presentation_for_result


def test_failure_analysis_problems_are_a_failed_agent_finding() -> None:
    view = presentation_for_result(
        {
            "status": "completed",
            "result": {
                "content": (
                    '{"summary": "Missing file.", "problems": ['
                    '{"problem": "No regex", "fault": "agent", "description": "Never wrote it."}'
                    '], "confidence": "high"}'
                )
            },
        }
    )

    assert view.kind == "problems"
    assert view.tag == "failed"
    assert view.answer == "Agent failure"
    assert view.findings[0].title == "No regex"
    assert view.findings[0].fault == "agent"


def test_failure_analysis_empty_problems_is_passed() -> None:
    view = presentation_for_result(
        {
            "status": "completed",
            "result": {"summary": "Clean.", "problems": [], "confidence": "high"},
        }
    )

    assert view.tag == "passed"
    assert view.answer == "No failure"
    assert view.findings == ()


def test_mixed_faults_are_labeled_mixed_failure() -> None:
    view = presentation_for_result(
        {
            "status": "completed",
            "result": {
                "problems": [
                    {"problem": "Bad regex", "fault": "agent"},
                    {"problem": "Cut off", "fault": "unclear"},
                ]
            },
        }
    )

    assert view.tag == "failed"
    assert view.answer == "Mixed failure"


def test_false_negative_yes_is_failed_without_findings() -> None:
    view = presentation_for_result(
        {
            "status": "completed",
            "result": {"content": '{"is_false_negative": true, "reasoning": "Grader missed it."}'},
        }
    )

    assert view.kind == "boolean"
    assert view.tag == "failed"
    assert view.label == "False Negative"
    assert view.answer == "yes"
    assert view.findings == ()
    assert view.summary == "Grader missed it."


def test_queued_runs_are_pending_not_passed() -> None:
    view = presentation_for_result({"status": "queued", "canonical_result": None})

    assert view.kind == "pending"
    assert view.tag == "unknown"
    assert view.label == "queued"


def test_standard_result_blob_detects_boolean_and_problems() -> None:
    assert is_standard_result_blob('{"is_false_negative": false}')
    assert is_standard_result_blob('{"summary": "x", "problems": []}')
    assert not is_standard_result_blob("Checking the workspace.")
    assert not is_standard_result_blob('{"notes": "still working"}')
