"""Parse standard QA-agent result blobs into review chrome."""

from __future__ import annotations

from hud.cli.qa_analysis import (
    QA_AGENT_RESULT_V1,
    is_standard_result_blob,
    presentation_for_result,
    to_qa_agent_result_v1,
)


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


def test_queued_rows_have_no_canonical_v1() -> None:
    assert to_qa_agent_result_v1({"status": "queued", "canonical_result": None}) is None


def test_boolean_blob_normalizes_to_v1() -> None:
    result = to_qa_agent_result_v1(
        {
            "status": "completed",
            "result": {
                "is_false_negative": True,
                "reasoning": "The grader rejected a valid answer.",
                "confidence": "high",
            },
        }
    )

    assert result == {
        "schema_version": QA_AGENT_RESULT_V1,
        "verdict": "failed",
        "summary": "The grader rejected a valid answer.",
        "findings": [],
        "metadata": {
            "kind": "boolean",
            "label": "False Negative",
            "is_false_negative": True,
            "confidence": "high",
            "answer": "yes",
        },
    }


def test_reward_hacking_blob_keeps_strategy_in_v1_metadata() -> None:
    result = to_qa_agent_result_v1(
        {
            "status": "completed",
            "result": {
                "is_reward_hacking": True,
                "hacking_strategy": "test_manipulation",
                "reasoning": "The agent rewrote the hidden tests.",
            },
        }
    )

    assert result is not None
    assert result["verdict"] == "failed"
    assert result["metadata"]["is_reward_hacking"] is True
    assert result["metadata"]["hacking_strategy"] == "test_manipulation"


def test_failure_analysis_problems_normalize_to_v1_findings() -> None:
    result = to_qa_agent_result_v1(
        {
            "status": "completed",
            "result": {
                "summary": "Missing file.",
                "problems": [
                    {
                        "problem": "No regex",
                        "fault": "agent",
                        "description": "Never wrote it.",
                    }
                ],
                "confidence": "high",
            },
        }
    )

    assert result == {
        "schema_version": QA_AGENT_RESULT_V1,
        "verdict": "failed",
        "summary": "Missing file.",
        "findings": [
            {
                "summary": "No regex",
                "description": "Never wrote it.",
                "fault": "agent",
            }
        ],
        "metadata": {
            "kind": "problems",
            "label": "Failure Analysis",
            "answer": "Agent failure",
            "confidence": "high",
        },
    }


def test_existing_v1_blob_is_returned_as_v1() -> None:
    result = to_qa_agent_result_v1(
        {
            "status": "completed",
            "canonical_result": {
                "schema_version": QA_AGENT_RESULT_V1,
                "verdict": "passed",
                "summary": "Looks good.",
                "findings": [],
                "metadata": {},
            },
        }
    )

    assert result == {
        "schema_version": QA_AGENT_RESULT_V1,
        "verdict": "passed",
        "summary": "Looks good.",
        "findings": [],
        "metadata": {"kind": "qa_result"},
    }
