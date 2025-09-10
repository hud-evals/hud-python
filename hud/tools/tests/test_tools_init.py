"""Tests for hud.tools.__init__ module."""

from __future__ import annotations

import pytest


class TestToolsInit:
    """Tests for the tools package initialization."""

    def test_lazy_import_anthropic_computer_tool(self):
        """Test lazy import of AnthropicComputerTool."""
        from hud.tools import AnthropicComputerTool

        # Verify it's imported correctly
        assert AnthropicComputerTool.__name__ == "AnthropicComputerTool"

    def test_lazy_import_hud_computer_tool(self):
        """Test lazy import of HudComputerTool."""
        from hud.tools import HudComputerTool

        # Verify it's imported correctly
        assert HudComputerTool.__name__ == "HudComputerTool"

    def test_lazy_import_openai_computer_tool(self):
        """Test lazy import of OpenAIComputerTool."""
        from hud.tools import OpenAIComputerTool

        # Verify it's imported correctly
        assert OpenAIComputerTool.__name__ == "OpenAIComputerTool"

    def test_lazy_import_invalid_attribute(self):
        """Test lazy import with invalid attribute name."""
        import hud.tools as tools_module

        with pytest.raises(AttributeError, match=r"module '.*' has no attribute 'InvalidTool'"):
            _ = tools_module.InvalidTool

    def test_direct_imports_available(self):
        """Test that directly imported tools are available."""
        from hud.tools import BaseHub, BaseTool, BashTool, EditTool, PlaywrightTool, ResponseTool

        # All should be available
        assert BaseHub is not None
        assert BaseTool is not None
        assert BashTool is not None
        assert EditTool is not None
        assert PlaywrightTool is not None
        assert ResponseTool is not None
