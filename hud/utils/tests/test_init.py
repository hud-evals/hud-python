"""Test utils package imports."""

from __future__ import annotations


def test_utils_imports():
    """Test that utils package can be imported."""
    import hud.utils

    # Check that the module exists
    assert hud.utils is not None

    # Try importing submodules
    from hud.utils import telemetry

    assert telemetry is not None
