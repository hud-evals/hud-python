"""Tests for settings module."""

from __future__ import annotations

from hud.settings import Settings, get_settings, settings


def test_get_settings():
    """Test that get_settings returns the singleton settings instance."""
    result = get_settings()
    assert isinstance(result, Settings)
    assert result is settings  # Should be the same singleton instance


def test_settings_defaults():
    """Test that settings have expected default values."""
    s = get_settings()
    assert s.base_url == "https://orchestration.hud.so/hud-gym/api"
    assert s.hud_mcp_url == "https://mcp.hud.so/v3/mcp"
    assert s.telemetry_enabled is True
    assert s.hud_logging is True
    assert s.log_stream == "stdout"


def test_settings_singleton():
    """Test that settings is a singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert s1 is settings
