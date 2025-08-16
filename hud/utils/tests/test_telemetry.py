from __future__ import annotations

from unittest.mock import patch

from hud.utils.telemetry import stream


def test_stream():
    html_content = stream("https://example.com")
    assert html_content is not None
    assert "<div style=" in html_content
    assert 'src="https://example.com"' in html_content


def test_stream_with_display_exception():
    """Test stream when IPython display raises an exception."""
    with (
        patch("IPython.display.display", side_effect=Exception("Display error")),
        patch("hud.utils.telemetry.logger") as mock_logger,
    ):
        html_content = stream("https://example.com")

        # Should still return the HTML content
        assert html_content is not None
        assert 'src="https://example.com"' in html_content

        # Should log the warning
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args[0]
        assert "Display error" in str(args[0])


def test_display_screenshot():
    from hud.utils.telemetry import display_screenshot

    # This is a simple 1x1 transparent PNG image in base64 format
    base64_image = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
        "AAABJRU5ErkJggg=="
    )

    html_content = display_screenshot(base64_image)
    assert html_content is not None
    assert "<div style=" in html_content
    assert "width: 960px" in html_content
    assert "height: 540px" in html_content
    assert f"data:image/png;base64,{base64_image}" in html_content

    # Test with custom dimensions
    custom_html = display_screenshot(base64_image, width=800, height=600)
    assert "width: 800px" in custom_html
    assert "height: 600px" in custom_html

    # Test with data URI already included
    data_uri = f"data:image/png;base64,{base64_image}"
    uri_html = display_screenshot(data_uri)
    assert data_uri in uri_html


def test_display_screenshot_with_exception():
    """Test display_screenshot when IPython display raises an exception."""
    from hud.utils.telemetry import display_screenshot

    base64_image = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
        "AAABJRU5ErkJggg=="
    )

    with (
        patch("IPython.display.display", side_effect=Exception("Display error")),
        patch("hud.utils.telemetry.logger") as mock_logger,
    ):
        html_content = display_screenshot(base64_image)

        # Should still return the HTML content
        assert html_content is not None
        assert f"data:image/png;base64,{base64_image}" in html_content

        # Should log the warning
        mock_logger.warning.assert_called_once()
        args = mock_logger.warning.call_args[0]
        assert "Display error" in str(args[0])
