"""Tests for hud.__init__ module."""

from __future__ import annotations

import sys
from unittest.mock import patch


class TestHudInit:
    """Tests for the hud package initialization."""

    def test_version_import_success(self):
        """Test that version is imported successfully."""
        import hud

        # Version should be available
        assert hasattr(hud, "__version__")
        assert isinstance(hud.__version__, str)
        assert hud.__version__ != "unknown"

    def test_version_import_fallback(self):
        """Test that version falls back to 'unknown' when import fails."""
        # Mock the version module to raise ImportError
        with patch.dict("sys.modules", {"hud.version": None}):
            # Remove hud from modules if it's already loaded to force reimport
            if "hud" in sys.modules:
                del sys.modules["hud"]

            # Now import should use fallback
            import hud

            # Should have the fallback version
            assert hud.__version__ == "unknown"

            # Clean up - remove the module so subsequent tests work
            if "hud" in sys.modules:
                del sys.modules["hud"]

    def test_all_exports_available(self):
        """Test that all exported functions are available."""
        import hud

        expected_exports = [
            "Environment",
            "EvalContext",
            "eval",
            "instrument",
        ]

        for export in expected_exports:
            assert hasattr(hud, export), f"Missing export: {export}"
