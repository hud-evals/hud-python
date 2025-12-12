"""Tests for settings module."""

from __future__ import annotations

from hud.settings import Settings, get_settings, settings


def test_get_settings():
    """Test that get_settings returns the singleton settings instance."""
    result = get_settings()
    assert isinstance(result, Settings)
    assert result is settings  # Should be the same singleton instance


def test_settings_defaults():
    """Test that settings have expected default values or env overrides."""
    s = get_settings()
    # These URLs may be overridden by environment variables
    assert s.hud_telemetry_url.endswith("/v3/api")
    assert s.hud_mcp_url.endswith("/v3/mcp")
    # Default may be overridden in CI; just assert the field exists and is bool
    assert isinstance(s.telemetry_enabled, bool)
    assert s.hud_logging is True
    assert s.log_stream == "stdout"


def test_settings_singleton():
    """Test that settings is a singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert s1 is settings
