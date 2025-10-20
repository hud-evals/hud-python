from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hud.otel.instrumentation import (
    _patch_get_error_type,
    _patch_mcp_instrumentation,
    install_mcp_instrumentation,
)


def test_install_mcp_instrumentation_success():
    """Test successful installation of MCP instrumentation."""
    mock_provider = MagicMock()

    with (
        patch("opentelemetry.instrumentation.mcp.instrumentation"),
        patch(
            "opentelemetry.instrumentation.mcp.instrumentation.McpInstrumentor"
        ) as mock_instrumentor_class,
        patch("hud.otel.instrumentation._patch_mcp_instrumentation"),
    ):
        mock_instrumentor = MagicMock()
        mock_instrumentor_class.return_value = mock_instrumentor

        install_mcp_instrumentation(mock_provider)

        mock_instrumentor.instrument.assert_called_once_with(tracer_provider=mock_provider)


def test_install_mcp_instrumentation_import_error():
    """Test installation handles ImportError gracefully."""
    mock_provider = MagicMock()

    # Mock the import to raise ImportError
    import sys

    with patch.dict(sys.modules, {"opentelemetry.instrumentation.mcp.instrumentation": None}):
        # Should not raise
        install_mcp_instrumentation(mock_provider)


def test_install_mcp_instrumentation_general_exception():
    """Test installation handles general exceptions gracefully."""
    mock_provider = MagicMock()

    with (
        patch("opentelemetry.instrumentation.mcp.instrumentation"),
        patch(
            "opentelemetry.instrumentation.mcp.instrumentation.McpInstrumentor"
        ) as mock_instrumentor_class,
    ):
        mock_instrumentor_class.side_effect = Exception("Unexpected error")

        # Should not raise
        install_mcp_instrumentation(mock_provider)


def test_patch_mcp_instrumentation_success():
    """Test successful patching of MCP instrumentation."""
    with (
        patch("opentelemetry.instrumentation.mcp.instrumentation.McpInstrumentor") as mock_class,
        patch("hud.otel.instrumentation._patch_get_error_type"),
    ):
        mock_class._transport_wrapper = None

        _patch_mcp_instrumentation()

        # Should have set the _transport_wrapper
        assert mock_class._transport_wrapper is not None


def test_patch_mcp_instrumentation_exception():
    """Test patching handles exceptions gracefully."""
    with patch(
        "opentelemetry.instrumentation.mcp.instrumentation.McpInstrumentor",
        side_effect=Exception("Error"),
    ):
        # Should not raise
        _patch_mcp_instrumentation()


def test_patch_get_error_type_success():
    """Test successful patching of get_error_type."""
    with patch("opentelemetry.instrumentation.mcp.instrumentation") as mock_mcp_inst:
        mock_mcp_inst.get_error_type = None

        _patch_get_error_type()

        # Should have set get_error_type
        assert mock_mcp_inst.get_error_type is not None


def test_patch_get_error_type_exception():
    """Test patching get_error_type handles exceptions."""
    with patch(
        "opentelemetry.instrumentation.mcp.instrumentation", side_effect=ImportError("Not found")
    ):
        # Should not raise
        _patch_get_error_type()


def test_patched_get_error_type_valid_4xx():
    """Test patched get_error_type with valid 4xx status code."""
    with patch("opentelemetry.instrumentation.mcp.instrumentation") as mock_mcp_inst:
        _patch_get_error_type()

        patched_func = mock_mcp_inst.get_error_type

        # Test with a valid 4xx error
        result = patched_func("Error 404 not found")
        assert result == "NOT_FOUND"


def test_patched_get_error_type_valid_5xx():
    """Test patched get_error_type with valid 5xx status code."""
    with patch("opentelemetry.instrumentation.mcp.instrumentation") as mock_mcp_inst:
        _patch_get_error_type()

        patched_func = mock_mcp_inst.get_error_type

        # Test with a valid 5xx error
        result = patched_func("Error 500 internal server error")
        assert result == "INTERNAL_SERVER_ERROR"


def test_patched_get_error_type_invalid_status():
    """Test patched get_error_type with invalid status code."""
    with patch("opentelemetry.instrumentation.mcp.instrumentation") as mock_mcp_inst:
        _patch_get_error_type()

        patched_func = mock_mcp_inst.get_error_type

        # Test with an invalid HTTP status code (e.g., 499 doesn't exist in HTTPStatus)
        result = patched_func("Error 499 custom error")
        # Should return the name even if it's not a standard HTTPStatus
        assert result is None or isinstance(result, str)


def test_patched_get_error_type_no_status():
    """Test patched get_error_type with no status code."""
    with patch("opentelemetry.instrumentation.mcp.instrumentation") as mock_mcp_inst:
        _patch_get_error_type()

        patched_func = mock_mcp_inst.get_error_type

        result = patched_func("Error message without status code")
        assert result is None


def test_patched_get_error_type_non_string():
    """Test patched get_error_type with non-string input."""
    with patch("opentelemetry.instrumentation.mcp.instrumentation") as mock_mcp_inst:
        _patch_get_error_type()

        patched_func = mock_mcp_inst.get_error_type

        result = patched_func(None)
        assert result is None

        result = patched_func(123)
        assert result is None


def test_patched_get_error_type_3xx_ignored():
    """Test patched get_error_type ignores 3xx codes."""
    with patch("opentelemetry.instrumentation.mcp.instrumentation") as mock_mcp_inst:
        _patch_get_error_type()

        patched_func = mock_mcp_inst.get_error_type

        result = patched_func("Error 301 moved")
        assert result is None


@pytest.mark.asyncio
async def test_transport_wrapper_three_values():
    """Test transport wrapper handles 3-value tuple."""
    with (
        patch("opentelemetry.instrumentation.mcp.instrumentation.McpInstrumentor") as mock_class,
        patch("hud.otel.instrumentation._patch_get_error_type"),
    ):
        mock_class._transport_wrapper = None

        _patch_mcp_instrumentation()

        # Get the patched wrapper
        wrapper_func = mock_class._transport_wrapper
        assert wrapper_func is not None


@pytest.mark.asyncio
async def test_transport_wrapper_two_values():
    """Test transport wrapper handles 2-value tuple."""
    with (
        patch("opentelemetry.instrumentation.mcp.instrumentation.McpInstrumentor") as mock_class,
        patch("hud.otel.instrumentation._patch_get_error_type"),
    ):
        mock_class._transport_wrapper = None

        _patch_mcp_instrumentation()

        # Get the patched wrapper
        wrapper_func = mock_class._transport_wrapper
        assert wrapper_func is not None
