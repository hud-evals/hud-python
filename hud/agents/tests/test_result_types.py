"""Agent/scenario result types in ``hud.agents.types``.

``ContentResult`` (combine + content blocks), ``SubScore``, ``ScenarioResult`` /
``EvaluationResult``, ``AgentAnswer``, ``Citation``, ``ToolError`` — pure data shapes.
"""

from __future__ import annotations

import pytest
from mcp.types import ImageContent, TextContent

from hud.agents.types import (
    AgentAnswer,
    Citation,
    ContentResult,
    EvaluationResult,
    ScenarioResult,
    SubScore,
    ToolError,
)

# ─── ContentResult ────────────────────────────────────────────────────


def test_content_result_concatenates_text_fields() -> None:
    combined = ContentResult(output="a", error="e1") + ContentResult(output="b", error="e2")
    assert combined.output == "ab"
    assert combined.error == "e1e2"


def test_content_result_takes_either_side_when_one_empty() -> None:
    combined = ContentResult(output="only") + ContentResult(error="err")
    assert combined.output == "only"
    assert combined.error == "err"


def test_content_result_rejects_combining_two_images() -> None:
    with pytest.raises(ValueError, match="Cannot combine"):
        _ = ContentResult(base64_image="a") + ContentResult(base64_image="b")


def test_content_result_text_blocks_include_url_marker() -> None:
    blocks = ContentResult(output="hi", url="https://example.com").to_text_blocks()
    texts = [b.text for b in blocks]
    assert "hi" in texts
    assert "__URL__:https://example.com" in texts


def test_content_result_image_block_detects_mime() -> None:
    png = ContentResult(base64_image="iVBORw0KGgo=").to_content_blocks()
    jpeg = ContentResult(base64_image="/9j/4AAQ").to_content_blocks()

    png_img = next(b for b in png if isinstance(b, ImageContent))
    jpeg_img = next(b for b in jpeg if isinstance(b, ImageContent))
    assert png_img.mimeType == "image/png"
    assert jpeg_img.mimeType == "image/jpeg"


def test_content_result_text_only_has_no_image_block() -> None:
    blocks = ContentResult(output="x").to_content_blocks()
    assert all(isinstance(b, TextContent) for b in blocks)


# ─── SubScore / EvaluationResult ──────────────────────────────────────


def test_subscore_score_aliases_value() -> None:
    s = SubScore(name="acc", value=0.75, weight=1.0)
    assert s.score == 0.75


def test_evaluation_result_from_float() -> None:
    r = EvaluationResult.from_float(0.25)
    assert r.reward == 0.25
    assert r.done is True


def test_evaluation_result_is_scenario_result_alias() -> None:
    assert EvaluationResult is ScenarioResult


def test_evaluation_result_warns_when_subscores_disagree_with_reward() -> None:
    with pytest.warns(UserWarning):
        EvaluationResult(reward=1.0, subscores=[SubScore(name="a", value=0.5, weight=1.0)])


# ─── AgentAnswer / Citation / ToolError ───────────────────────────────


def test_agent_answer_holds_parsed_content_and_citations() -> None:
    answer = AgentAnswer(
        content={"final": "42"},
        raw='{"final": "42"}',
        citations=[Citation(type="url_citation", source="https://x", text="span")],
    )
    assert answer.content == {"final": "42"}
    assert answer.raw == '{"final": "42"}'
    assert answer.citations[0].source == "https://x"


def test_citation_defaults() -> None:
    c = Citation()
    assert c.type == "citation"
    assert c.text == ""
    assert c.start_index is None


def test_tool_error_is_an_exception() -> None:
    assert issubclass(ToolError, Exception)
    with pytest.raises(ToolError, match="boom"):
        raise ToolError("boom")
