"""Tests for hud.cli.__main__ module."""

from __future__ import annotations

import pytest


class TestCLIMain:
    """Test the __main__ module."""

    def test_main_module_exists(self) -> None:
        """Test that __main__.py exists and can be imported."""
        # Just verify the module can be imported
        import hud.cli.__main__

        assert hud.cli.__main__ is not None

    def test_main_module_has_main_import(self) -> None:
        """Test that __main__.py imports main from the package."""
        import hud.cli.__main__

        # The module should have imported main
        assert hasattr(hud.cli.__main__, "main")


if __name__ == "__main__":
    pytest.main([__file__])
