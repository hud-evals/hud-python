from __future__ import annotations

from importlib.metadata import version


def test_import():
    """Test that the package can be imported."""
    import hud

    assert hud.__version__ == version("hud")
