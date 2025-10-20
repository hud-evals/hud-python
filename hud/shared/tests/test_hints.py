from __future__ import annotations

from unittest.mock import MagicMock, patch

from hud.shared.hints import (
    CLIENT_NOT_INITIALIZED,
    ENV_VAR_MISSING,
    HUD_API_KEY_MISSING,
    INVALID_CONFIG,
    MCP_SERVER_ERROR,
    RATE_LIMIT_HIT,
    TOOL_NOT_FOUND,
    Hint,
    render_hints,
)


def test_hint_objects_basic():
    assert HUD_API_KEY_MISSING.title and isinstance(HUD_API_KEY_MISSING.tips, list)
    assert RATE_LIMIT_HIT.code == "RATE_LIMIT"
    assert TOOL_NOT_FOUND.title.startswith("Tool")
    assert CLIENT_NOT_INITIALIZED.message
    assert ENV_VAR_MISSING.command_examples is not None


def test_all_hint_constants():
    """Test that all predefined hint constants have required fields."""
    hints = [
        HUD_API_KEY_MISSING,
        RATE_LIMIT_HIT,
        TOOL_NOT_FOUND,
        CLIENT_NOT_INITIALIZED,
        INVALID_CONFIG,
        ENV_VAR_MISSING,
        MCP_SERVER_ERROR,
    ]

    for hint in hints:
        assert hint.title
        assert hint.message
        assert hint.code


def test_hint_creation():
    """Test creating a custom Hint."""
    hint = Hint(
        title="Test Hint",
        message="This is a test",
        tips=["Tip 1", "Tip 2"],
        docs_url="https://example.com",
        command_examples=["command 1"],
        code="TEST_CODE",
        context=["test", "custom"],
    )

    assert hint.title == "Test Hint"
    assert hint.message == "This is a test"
    assert hint.tips and len(hint.tips) == 2
    assert hint.docs_url == "https://example.com"
    assert hint.command_examples and len(hint.command_examples) == 1
    assert hint.code == "TEST_CODE"
    assert hint.context and "test" in hint.context


def test_hint_minimal():
    """Test creating a minimal Hint with only required fields."""
    hint = Hint(title="Minimal", message="Just basics")

    assert hint.title == "Minimal"
    assert hint.message == "Just basics"
    assert hint.tips is None
    assert hint.docs_url is None
    assert hint.command_examples is None
    assert hint.code is None
    assert hint.context is None


def test_render_hints_none():
    """Test that render_hints handles None gracefully."""
    # Should not raise
    render_hints(None)


def test_render_hints_empty_list():
    """Test that render_hints handles empty list gracefully."""
    # Should not raise
    render_hints([])


@patch("hud.utils.hud_console.hud_console")
def test_render_hints_with_tips(mock_console):
    """Test rendering hints with tips."""
    render_hints([HUD_API_KEY_MISSING])

    # Should call warning for title/message
    mock_console.warning.assert_called()
    # Should call info for tips
    assert mock_console.info.call_count >= 1


@patch("hud.utils.hud_console.hud_console")
def test_render_hints_with_command_examples(mock_console):
    """Test rendering hints with command examples."""
    render_hints([ENV_VAR_MISSING])

    # Should call command_example
    mock_console.command_example.assert_called()


@patch("hud.utils.hud_console.hud_console")
def test_render_hints_with_docs_url(mock_console):
    """Test rendering hints with documentation URL."""
    hint = Hint(
        title="Test",
        message="Test message",
        docs_url="https://docs.example.com",
    )

    render_hints([hint])

    # Should call link for docs URL
    mock_console.link.assert_called_with("https://docs.example.com")


@patch("hud.utils.hud_console.hud_console")
def test_render_hints_same_title_and_message(mock_console):
    """Test rendering hints when title equals message."""
    hint = Hint(title="Same", message="Same")

    render_hints([hint])

    # Should only call warning once with just the message
    mock_console.warning.assert_called_once_with("Same")


@patch("hud.utils.hud_console.hud_console")
def test_render_hints_different_title_and_message(mock_console):
    """Test rendering hints when title differs from message."""
    hint = Hint(title="Title", message="Different message")

    render_hints([hint])

    # Should call warning with both title and message
    mock_console.warning.assert_called_once()
    call_args = mock_console.warning.call_args[0][0]
    assert "Title" in call_args
    assert "Different message" in call_args


def test_render_hints_with_custom_design():
    """Test rendering hints with custom design object."""
    custom_design = MagicMock()

    hint = Hint(title="Test", message="Message")
    # Should not raise when custom design is provided
    render_hints([hint], design=custom_design)


@patch("hud.utils.hud_console.hud_console")
def test_render_hints_handles_exception(mock_console):
    """Test that render_hints handles exceptions gracefully."""
    mock_console.warning.side_effect = Exception("Test error")

    hint = Hint(title="Test", message="Message")

    # Should not raise, just log warning
    render_hints([hint])
