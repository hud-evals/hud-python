from __future__ import annotations

import pytest
from mcp.types import ImageContent, TextContent

from hud.tools.types import ContentResult, EvaluationResult, SubScore, ToolError


def test_evaluation_result_defaults():
    """Test EvaluationResult with default values."""
    result = EvaluationResult()

    assert result.reward == 0.0
    assert result.done is True  # Default is True (task complete)
    assert result.content is None
    assert result.info == {}
    assert result.isError is False


def test_evaluation_result_with_values():
    """Test EvaluationResult with custom values."""
    result = EvaluationResult(
        reward=0.95,
        done=True,
        content="Task completed successfully",
        info={"steps": 5},
        isError=False,
    )

    assert result.reward == 0.95
    assert result.done is True
    assert result.content == "Task completed successfully"
    assert result.info == {"steps": 5}
    assert result.isError is False


def test_content_result_defaults():
    """Test ContentResult with default values."""
    result = ContentResult()

    assert result.output is None
    assert result.error is None
    assert result.base64_image is None
    assert result.system is None


def test_content_result_with_values():
    """Test ContentResult with custom values."""
    result = ContentResult(
        output="Command executed",
        error="No errors",
        base64_image="base64data",
        system="System message",
    )

    assert result.output == "Command executed"
    assert result.error == "No errors"
    assert result.base64_image == "base64data"
    assert result.system == "System message"


def test_content_result_add_both_output():
    """Test adding two ContentResults with output."""
    result1 = ContentResult(output="Part 1")
    result2 = ContentResult(output=" Part 2")

    combined = result1 + result2

    assert combined.output == "Part 1 Part 2"
    assert combined.error is None
    assert combined.base64_image is None


def test_content_result_add_both_error():
    """Test adding two ContentResults with errors."""
    result1 = ContentResult(error="Error 1")
    result2 = ContentResult(error=" Error 2")

    combined = result1 + result2

    assert combined.error == "Error 1 Error 2"
    assert combined.output is None


def test_content_result_add_both_system():
    """Test adding two ContentResults with system messages."""
    result1 = ContentResult(system="System 1")
    result2 = ContentResult(system=" System 2")

    combined = result1 + result2

    assert combined.system == "System 1 System 2"


def test_content_result_add_one_sided():
    """Test adding ContentResults where only one has values."""
    result1 = ContentResult(output="Output")
    result2 = ContentResult(error="Error")

    combined = result1 + result2

    assert combined.output == "Output"
    assert combined.error == "Error"


def test_content_result_add_images_raises_error():
    """Test that combining two results with images raises an error."""
    result1 = ContentResult(base64_image="image1")
    result2 = ContentResult(base64_image="image2")

    with pytest.raises(ValueError, match="Cannot combine tool results"):
        _ = result1 + result2


def test_content_result_add_one_image():
    """Test adding ContentResults where only one has an image."""
    result1 = ContentResult(base64_image="image1")
    result2 = ContentResult(output="Output")

    combined = result1 + result2

    assert combined.base64_image == "image1"
    assert combined.output == "Output"


def test_content_result_to_content_blocks_output():
    """Test converting ContentResult with output to content blocks."""
    result = ContentResult(output="Test output")

    blocks = result.to_content_blocks()

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "Test output"


def test_content_result_to_content_blocks_error():
    """Test converting ContentResult with error to content blocks."""
    result = ContentResult(error="Test error")

    blocks = result.to_content_blocks()

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "Test error"


def test_content_result_to_content_blocks_image():
    """Test converting ContentResult with image to content blocks."""
    result = ContentResult(base64_image="base64data")

    blocks = result.to_content_blocks()

    assert len(blocks) == 1
    assert isinstance(blocks[0], ImageContent)
    assert blocks[0].data == "base64data"
    assert blocks[0].mimeType == "image/png"


def test_content_result_to_content_blocks_all():
    """Test converting ContentResult with all fields to content blocks."""
    result = ContentResult(
        output="Output",
        error="Error",
        base64_image="image",
    )

    blocks = result.to_content_blocks()

    assert len(blocks) == 3
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "Output"
    assert isinstance(blocks[1], TextContent)
    assert blocks[1].text == "Error"
    assert isinstance(blocks[2], ImageContent)
    assert blocks[2].data == "image"


def test_content_result_to_content_blocks_empty():
    """Test converting empty ContentResult to content blocks."""
    result = ContentResult()

    blocks = result.to_content_blocks()

    assert len(blocks) == 0


def test_tool_error():
    """Test ToolError exception."""
    error = ToolError("Test error message")

    assert isinstance(error, Exception)
    assert str(error) == "Test error message"


def test_subscore_basic():
    """Test SubScore with required fields."""
    subscore = SubScore(name="accuracy", value=0.85)

    assert subscore.name == "accuracy"
    assert subscore.weight == 1.0  # Default
    assert subscore.value == 0.85


def test_subscore_with_weight():
    """Test SubScore with custom weight."""
    subscore = SubScore(name="speed", weight=0.3, value=0.9)

    assert subscore.name == "speed"
    assert subscore.weight == 0.3
    assert subscore.value == 0.9


def test_evaluation_result_with_subscores():
    """Test EvaluationResult with subscores."""
    result = EvaluationResult(
        reward=0.82,
        done=True,
        subscores=[
            SubScore(name="accuracy", weight=0.6, value=0.9),
            SubScore(name="speed", weight=0.4, value=0.7),
        ],
    )

    assert result.reward == 0.82
    assert result.subscores is not None
    assert len(result.subscores) == 2
    assert result.subscores[0].name == "accuracy"
    assert result.subscores[1].name == "speed"


def test_evaluation_result_from_float():
    """Test EvaluationResult.from_float() convenience method."""
    result = EvaluationResult.from_float(0.75)

    assert result.reward == 0.75
    assert result.done is True
    assert result.content is None
    assert result.subscores is None


def test_evaluation_result_model_dump():
    """Test that EvaluationResult serializes correctly."""
    result = EvaluationResult(
        reward=0.9,
        done=True,
        content="Test content",
        subscores=[SubScore(name="test", value=0.9)],
    )

    data = result.model_dump(exclude_none=True)

    assert data["reward"] == 0.9
    assert data["done"] is True
    assert data["content"] == "Test content"
    assert len(data["subscores"]) == 1
    assert data["subscores"][0]["name"] == "test"


# Tests for ContentResult.to_text_blocks()


def test_content_result_to_text_blocks_output():
    """Test to_text_blocks with output only."""
    from mcp.types import TextContent

    result = ContentResult(output="Hello world")
    blocks = result.to_text_blocks()

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "Hello world"


def test_content_result_to_text_blocks_error():
    """Test to_text_blocks with error."""
    from mcp.types import TextContent

    result = ContentResult(error="Something went wrong")
    blocks = result.to_text_blocks()

    assert len(blocks) == 1
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "Something went wrong"


def test_content_result_to_text_blocks_with_url():
    """Test to_text_blocks includes URL marker."""
    from mcp.types import TextContent

    result = ContentResult(output="Result", url="https://example.com")
    blocks = result.to_text_blocks()

    assert len(blocks) == 2
    assert isinstance(blocks[0], TextContent)
    assert blocks[0].text == "Result"
    assert isinstance(blocks[1], TextContent)
    assert "__URL__:https://example.com" in blocks[1].text


def test_content_result_to_text_blocks_all():
    """Test to_text_blocks with all text fields."""
    from mcp.types import TextContent

    result = ContentResult(
        output="Output text",
        error="Error text",
        url="https://example.com",
    )
    blocks = result.to_text_blocks()

    assert len(blocks) == 3
    assert all(isinstance(b, TextContent) for b in blocks)


def test_content_result_to_text_blocks_excludes_image():
    """Test to_text_blocks does NOT include base64_image."""
    result = ContentResult(
        output="Text output",
        base64_image="iVBORw0KGgo...",  # Fake base64
    )
    blocks = result.to_text_blocks()

    # Should only have the text block, not the image
    assert len(blocks) == 1
    assert blocks[0].text == "Text output"


def test_content_result_to_text_blocks_empty():
    """Test to_text_blocks with empty ContentResult."""
    result = ContentResult()
    blocks = result.to_text_blocks()

    assert len(blocks) == 0


# Tests for EvaluationResult default done=True


def test_evaluation_result_done_defaults_true():
    """Test that done defaults to True."""
    result = EvaluationResult(reward=0.5)
    assert result.done is True


def test_evaluation_result_from_float_done_true():
    """Test from_float sets done=True."""
    result = EvaluationResult.from_float(0.75)
    assert result.done is True


def test_evaluation_result_explicit_done_false():
    """Test done can be explicitly set to False."""
    result = EvaluationResult(reward=0.25, done=False)
    assert result.done is False


# Tests for SubScore validation


def test_subscore_metadata_excluded_from_dump():
    """Test SubScore.metadata is excluded from model_dump."""
    s = SubScore(name="x", value=1.0, metadata={"k": "v"})
    d = s.model_dump()
    assert "metadata" not in d
    assert s.metadata == {"k": "v"}


def test_subscore_metadata_none_by_default():
    """Test SubScore.metadata defaults to None."""
    s = SubScore(name="x", value=0.5)
    assert s.metadata is None


def test_subscore_score_alias():
    """Test SubScore.score returns same as .value."""
    s = SubScore(name="x", value=0.8)
    assert s.score == 0.8
    assert s.score == s.value


def test_subscore_forbids_extra_fields():
    """Test SubScore rejects extra fields."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SubScore(name="test", value=0.5, extra_field="not allowed")  # type: ignore[call-arg]


def test_subscore_requires_name():
    """Test SubScore requires name."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SubScore(value=0.5)  # type: ignore[call-arg]  # Missing name


def test_subscore_requires_value():
    """Test SubScore requires value."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SubScore(name="test")  # type: ignore[call-arg]  # Missing value


# Tests for EvaluationResult with info dict


def test_evaluation_result_info_dict():
    """Test EvaluationResult with info metadata."""
    result = EvaluationResult(
        reward=0.8,
        info={"steps": 10, "tokens": 500, "model": "gpt-4"},
    )

    assert result.info["steps"] == 10
    assert result.info["tokens"] == 500
    assert result.info["model"] == "gpt-4"


def test_evaluation_result_info_defaults_empty():
    """Test info defaults to empty dict."""
    result = EvaluationResult(reward=0.5)
    assert result.info == {}


def test_evaluation_result_isError_flag():
    """Test isError flag for failed evaluations."""
    result = EvaluationResult(
        reward=0.0,
        isError=True,
        content="Evaluation failed due to timeout",
    )

    assert result.isError is True
    assert result.reward == 0.0


# Tests for SubScore and EvaluationResult validators


def test_subscore_value_range_rejected():
    """Test SubScore rejects values outside [0, 1]."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SubScore(name="test", value=-0.1)
    with pytest.raises(ValidationError):
        SubScore(name="test", value=1.5)


def test_check_subscores_duplicate_names_warns():
    """Test duplicate subscore names produce a warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EvaluationResult(
            reward=0.5,
            subscores=[
                SubScore(name="accuracy", weight=0.5, value=0.5),
                SubScore(name="accuracy", weight=0.5, value=0.5),
            ],
        )
    assert any("Duplicate subscore names" in str(x.message) for x in w)


def test_check_subscores_weights_not_summing_to_one_warns():
    """Test positive weights not summing to ~1.0 produce a warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EvaluationResult(
            reward=0.75,
            subscores=[
                SubScore(name="a", weight=0.5, value=1.0),
                SubScore(name="b", weight=0.25, value=1.0),
            ],
        )
    assert any("Positive subscore weights should sum to ~1.0" in str(x.message) for x in w)


def test_check_subscores_reward_mismatch_warns():
    """Test weighted sum not matching reward produces a warning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EvaluationResult(
            reward=0.5,
            subscores=[SubScore(name="a", weight=1.0, value=0.8)],
        )
    assert any("Subscores don't match reward" in str(x.message) for x in w)


def test_check_subscores_valid_with_negative_weights():
    """Test valid subscores with negative weights produce no warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Positive: 0.6 + 0.4 = 1.0
        # Weighted sum: 0.6*1.0 + 0.4*0.5 + (-0.2)*1.0 = 0.6
        EvaluationResult(
            reward=0.6,
            subscores=[
                SubScore(name="quality", weight=0.6, value=1.0),
                SubScore(name="speed", weight=0.4, value=0.5),
                SubScore(name="penalty", weight=-0.2, value=1.0),
            ],
        )
    assert len(w) == 0
