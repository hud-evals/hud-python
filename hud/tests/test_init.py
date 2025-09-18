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
            "clear_trace",
            "create_job",
            "get_trace",
            "instrument",
            "job",
            "trace",
        ]

        for export in expected_exports:
            assert hasattr(hud, export), f"Missing export: {export}"

    def test_pretty_errors_import_and_install_success(self):
        """Test that pretty_errors module can be imported and install succeeds."""
        # Remove hud from modules to force reimport
        if "hud" in sys.modules:
            del sys.modules["hud"]

        # Import should work without issues
        import hud

        # Package should be importable and have expected attributes
        assert hasattr(hud, "__version__")
        assert hasattr(hud, "clear_trace")

        # Verify that pretty_errors module is available (was imported successfully)
        from hud.utils import pretty_errors

        assert hasattr(pretty_errors, "install_pretty_errors")

        # Clean up
        if "hud" in sys.modules:
            del sys.modules["hud"]

    def test_pretty_errors_install_failure_handled(self):
        """Test that package import handles install_pretty_errors failure gracefully."""
        # Mock the install_pretty_errors function to raise an exception
        original_install = None
        pretty_errors_module = None
        try:
            from hud.utils import pretty_errors as pe

            pretty_errors_module = pe
            original_install = pe.install_pretty_errors

            def failing_install():
                raise Exception("Install failed")

            pe.install_pretty_errors = failing_install

            # Remove hud from modules to force reimport
            if "hud" in sys.modules:
                del sys.modules["hud"]

            # Import should not raise exception despite install_pretty_errors failure
            import hud

            # Package should still be importable and have expected attributes
            assert hasattr(hud, "__version__")
            assert hasattr(hud, "clear_trace")

            # Clean up
            if "hud" in sys.modules:
                del sys.modules["hud"]

        finally:
            # Restore original function
            if original_install and pretty_errors_module:
                pretty_errors_module.install_pretty_errors = original_install

    def test_pretty_errors_import_failure_handled(self):
        """Test that package import handles pretty_errors import failure gracefully."""
        # Mock the pretty_errors module to raise ImportError
        with patch.dict("sys.modules", {"hud.utils.pretty_errors": None}):
            # Remove hud from modules if it's already loaded to force reimport
            if "hud" in sys.modules:
                del sys.modules["hud"]

            # Import should not raise exception despite pretty_errors import failure
            import hud

            # Package should still be importable and have expected attributes
            assert hasattr(hud, "__version__")
            assert hasattr(hud, "clear_trace")

            # Clean up - remove the module so subsequent tests work
            if "hud" in sys.modules:
                del sys.modules["hud"]
