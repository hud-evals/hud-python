"""Test tools package imports."""

from __future__ import annotations


def test_tools_imports():
    """Test that tools package can be imported."""
    import hud.tools

    # Check that the module exists
    assert hud.tools is not None

    # Try importing key submodules
    from hud.tools import base, coding, utils

    assert base is not None
    assert coding is not None
    assert utils is not None

    # Check key classes/functions
    assert hasattr(base, "BaseTool")
    assert hasattr(base, "BaseHub")
    assert hasattr(coding, "BashTool")
    assert hasattr(coding, "EditTool")
    assert hasattr(utils, "run")
    assert hasattr(utils, "maybe_truncate")
