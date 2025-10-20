from __future__ import annotations

import pytest
from mcp.types import ImageContent, TextContent

from hud.tools.types import ContentResult, EvaluationResult, ToolError


def test_evaluation_result_defaults():
    """Test EvaluationResult with default values."""
    result = EvaluationResult()

    assert result.reward == 0.0
    assert result.done is False
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
