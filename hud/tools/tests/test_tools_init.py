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
        from hud.tools import BaseHub, BaseTool, BashTool, EditTool, PlaywrightTool, SubmitTool

        # All should be available
        assert BaseHub is not None
        assert BaseTool is not None
        assert BashTool is not None
        assert EditTool is not None
        assert PlaywrightTool is not None
        assert SubmitTool is not None

    def test_filesystem_legacy_shims_register_base_primitives(self):
        """Legacy filesystem names construct canonical base primitives."""
        import hud.tools.filesystem as filesystem
        from hud.tools import GlobTool, GrepTool, ListTool, ReadTool

        read = ReadTool(base_path=".")
        grep = GrepTool(base_path=".")
        glob = GlobTool(base_path=".")
        listing = ListTool(base_path=".")

        assert isinstance(read, filesystem.ReadTool)
        assert isinstance(grep, filesystem.GrepTool)
        assert isinstance(glob, filesystem.GlobTool)
        assert isinstance(listing, filesystem.ListTool)
        assert read.name == "read"
        assert grep.name == "grep"
        assert glob.name == "glob"
        assert listing.name == "list"

    def test_gemini_filesystem_legacy_shims_register_base_primitives(self):
        """Legacy Gemini filesystem names construct canonical base primitives."""
        import hud.tools.filesystem as filesystem
        from hud.tools import (
            GeminiGlobTool,
            GeminiListTool,
            GeminiReadManyTool,
            GeminiReadTool,
            GeminiSearchTool,
        )

        read = GeminiReadTool(base_path=".")
        read_many = GeminiReadManyTool(base_path=".")
        search = GeminiSearchTool(base_path=".")
        glob = GeminiGlobTool(base_path=".")
        listing = GeminiListTool(base_path=".")

        assert isinstance(read, filesystem.ReadTool)
        assert isinstance(read_many, filesystem.ReadTool)
        assert isinstance(search, filesystem.GrepTool)
        assert isinstance(glob, filesystem.GlobTool)
        assert isinstance(listing, filesystem.ListTool)
        assert read.name == "read"
        assert read_many.name == "read"
        assert search.name == "grep"
        assert glob.name == "glob"
        assert listing.name == "list"

    def test_gemini_memory_legacy_shim_registers_memory_primitive(self):
        """Legacy Gemini memory name constructs the canonical memory primitive."""
        from hud.tools import GeminiMemoryTool
        from hud.tools.memory import MemoryTool

        memory = GeminiMemoryTool(memory_dir=".")

        assert isinstance(memory, MemoryTool)
        assert memory.name == "memory"
