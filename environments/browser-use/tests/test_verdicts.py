"""Keep structured grading judgments consistent with the final rubric score."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.parametrize(
    "response,verdict,explanation",
    [
        pytest.param(
            'Evaluation:\n```json\n{"criterion_status":"UNMET","explanation":"Evidence missing."}\n```\nVERDICT: MET',
            "UNMET",
            "Evidence missing.",
            id="embedded-unmet-overrides-trailing-met",
        ),
        pytest.param(
            'Evaluation:\n```json\n{"criterion_status":"MET","reason":"Evidence found."}\n```\nVERDICT: UNMET',
            "MET",
            "Evidence found.",
            id="embedded-met-overrides-trailing-unmet",
        ),
        ('{"criterion_status":"MET","explanation":"Evidence found."}', "MET", "Evidence found."),
        ('{"criterion_status":"UNMET","explanation":"Evidence missing."}', "UNMET", "Evidence missing."),
        ('{"status":"UNMET","reason":"Evidence missing."}', "UNMET", "Evidence missing."),
        ('{"verdict":"MET","reasoning":"Evidence found."}', "MET", "Evidence found."),
        ("Evidence found.\nVERDICT: MET", "MET", "Evidence found."),
        ("Evidence missing.\nVERDICT: UNMET", "UNMET", "Evidence missing."),
    ],
)
async def test_rubric_keeps_provider_verdict_and_explanation(browser_env, monkeypatch, response, verdict, explanation):
    import grader

    create = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(text=response)]))
    monkeypatch.setattr(
        grader, "AsyncAnthropic", lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(grader.settings, "api_key", "local-mocked-key")

    result = await grader.grade_with_rubric(
        "Research answer",
        [{"requirement": "Provides the requested evidence", "weight": 1}],
        "Research the question",
    )

    assert result["score"] == (1.0 if verdict == "MET" else 0.0)
    assert result["report"][0]["verdict"] == verdict
    assert result["report"][0]["reason"] == explanation
    create.assert_awaited_once()
