"""Test tools package imports."""

from __future__ import annotations


def test_tools_imports():
    """Test that tools package can be imported."""
    import hud.tools

    # Check that the module exists
    assert hud.tools is not None

    # Try importing key submodules
    from hud.tools import base, utils
    from hud.tools.coding import bash, edit

    assert base is not None
    assert bash is not None
    assert edit is not None
    assert utils is not None

    # Check key classes/functions
    assert hasattr(base, "BaseTool")
    assert hasattr(base, "BaseHub")
    assert hasattr(bash, "BashTool")
    assert hasattr(edit, "EditTool")
    assert hasattr(utils, "run")
    assert hasattr(utils, "maybe_truncate")
