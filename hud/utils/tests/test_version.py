from __future__ import annotations


def test_import():
    """Test that the package can be imported."""
    import hud

    assert hud.__version__ == "0.4.73"
