"""Tests for MCP utility functions."""

from __future__ import annotations

import pytest

from hud.utils.mcp import MCPConfigPatch, patch_mcp_config, setup_hud_telemetry


class TestPatchMCPConfig:
    """Tests for patch_mcp_config function."""

    def test_patch_headers_for_hud_servers(self):
        """Test patching headers for HUD MCP servers."""
        from hud.settings import get_settings

        settings = get_settings()

        # Create an MCP config with a HUD server URL
        mcp_config = {"test_server": {"url": f"{settings.hud_mcp_url}/test"}}

        # Create patch with headers
        patch = MCPConfigPatch(headers={"X-Test-Header": "test-value"}, meta=None)

        # Apply patch
        patch_mcp_config(mcp_config, patch)

        # Verify headers were added
        assert "headers" in mcp_config["test_server"]
        assert mcp_config["test_server"]["headers"]["X-Test-Header"] == "test-value"  # type: ignore[index]

    def test_patch_headers_preserves_existing(self):
        """Test that existing headers are preserved."""
        from hud.settings import get_settings

        settings = get_settings()

        # Create config with existing headers
        mcp_config = {
            "test_server": {
                "url": f"{settings.hud_mcp_url}/test",
                "headers": {"Existing-Header": "existing-value"},
            }
        }

        patch = MCPConfigPatch(
            headers={"X-Test-Header": "test-value", "Existing-Header": "new-value"},
            meta=None,
        )

        patch_mcp_config(mcp_config, patch)

        # Existing header should be preserved, new one added
        assert mcp_config["test_server"]["headers"]["Existing-Header"] == "existing-value"
        assert mcp_config["test_server"]["headers"]["X-Test-Header"] == "test-value"

    def test_patch_meta_for_all_servers(self):
        """Test patching metadata for all servers."""
        mcp_config = {
            "server1": {"url": "http://example.com"},
            "server2": {"url": "http://other.com"},
        }

        patch = MCPConfigPatch(headers=None, meta={"test_key": "test_value"})

        patch_mcp_config(mcp_config, patch)

        # Meta should be added to both servers
        assert mcp_config["server1"]["meta"]["test_key"] == "test_value"  # type: ignore[index]
        assert mcp_config["server2"]["meta"]["test_key"] == "test_value"  # type: ignore[index]

    def test_patch_meta_preserves_existing(self):
        """Test that existing meta is preserved."""
        mcp_config = {
            "test_server": {"url": "http://example.com", "meta": {"existing_key": "existing_value"}}
        }

        patch = MCPConfigPatch(
            headers=None,
            meta={"test_key": "test_value", "existing_key": "new_value"},
        )

        patch_mcp_config(mcp_config, patch)

        # Existing meta should be preserved, new one added
        assert mcp_config["test_server"]["meta"]["existing_key"] == "existing_value"
        assert mcp_config["test_server"]["meta"]["test_key"] == "test_value"


class TestSetupHUDTelemetry:
    """Tests for setup_hud_telemetry function."""

    def test_empty_config_returns_none(self):
        """Test that empty config returns None (no servers to set up telemetry for)."""
        result = setup_hud_telemetry({})
        assert result is None

    def test_none_config_raises_error(self):
        """Test that None config raises ValueError."""
        with pytest.raises(
            ValueError, match="Please run initialize\\(\\) before setting up client-side telemetry"
        ):
            setup_hud_telemetry(None)  # type: ignore[arg-type]

    def test_valid_config_returns_none_when_no_hud_servers(self):
        """Test that valid config with no HUD servers returns None."""
        mcp_config = {"test_server": {"url": "http://example.com"}}

        result = setup_hud_telemetry(mcp_config)
        assert result is None
