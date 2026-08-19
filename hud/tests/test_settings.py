"""Tests for settings module."""

from __future__ import annotations

from hud.settings import Settings, get_settings, settings


def test_get_settings():
    """Test that get_settings returns the singleton settings instance."""
    result = get_settings()
    assert isinstance(result, Settings)
    assert result is settings  # Should be the same singleton instance


def test_service_url_defaults():
    expected = {
        "hud_telemetry_url": "https://telemetry.hud.ai/v3/api",
        "hud_api_url": "https://api.hud.ai",
        "hud_web_url": "https://hud.ai",
        "hud_gateway_url": "https://inference.hud.ai",
        "hud_runtime_url": "https://mcp.hud.ai",
        "hud_rl_url": "https://rl.hud.ai",
    }

    for name, default in expected.items():
        assert Settings.model_fields[name].default == default


def test_file_tracking_is_enabled_by_default():
    assert Settings.model_fields["file_tracking_enabled"].default is True


def test_file_tracking_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HUD_FILE_TRACKING_ENABLED", "false")

    assert Settings().file_tracking_enabled is False


def test_cli_analytics_is_independent_of_trace_telemetry(monkeypatch):
    monkeypatch.setenv("HUD_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("HUD_CLI_ANALYTICS_ENABLED", "false")
    configured = Settings()

    assert configured.telemetry_enabled is True
    assert configured.cli_analytics_enabled is False


def test_settings_singleton():
    """Test that settings is a singleton."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert s1 is settings
