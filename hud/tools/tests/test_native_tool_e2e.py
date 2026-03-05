"""End-to-end tests for native tool spec propagation through MCP.

These tests verify that:
1. Tools with native_specs correctly embed meta data when served via MCPServer
2. Agents can retrieve and parse these native specs from MCP tools
3. Role-based exclusion works correctly end-to-end
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import suppress
from typing import Any, cast

import pytest
from fastmcp import Client as MCPClient

from hud.agents.base import MCPAgent
from hud.server import MCPServer
from hud.tools.coding import BashTool
from hud.tools.computer.anthropic import AnthropicComputerTool
from hud.tools.hosted import GoogleSearchTool
from hud.tools.native_types import NativeToolSpec
from hud.types import AgentType, InferenceResult


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _start_http_server(mcp: MCPServer, port: int) -> asyncio.Task[None]:
    task = asyncio.create_task(
        mcp.run_async(
            transport="http",
            host="127.0.0.1",
            port=port,
            path="/mcp",
            log_level="ERROR",
            show_banner=False,
        )
    )
    await asyncio.sleep(0.05)
    return task


class TestNativeToolSpecE2E:
    """Test native tool specs are properly transmitted via MCP."""

    @pytest.mark.asyncio
    async def test_bash_tool_meta_transmitted(self) -> None:
        """Test that BashTool's native_specs are transmitted via MCP meta field."""
        port = _free_port()
        mcp = MCPServer(name="BashToolTest")

        # Register BashTool which has native_specs for Claude
        bash_tool = BashTool()
        mcp.add_tool(bash_tool)

        server_task = await _start_http_server(mcp, port)

        try:
            cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
            client = MCPClient({"mcpServers": cfg})
            await client.__aenter__()

            tools = await client.list_tools()
            bash_tools = [t for t in tools if t.name == "bash"]
            assert len(bash_tools) == 1

            tool = bash_tools[0]
            assert tool.meta is not None, "Tool should have meta field"
            assert "native_tools" in tool.meta, "Meta should contain native_tools"
            assert "claude" in tool.meta["native_tools"], "Should have Claude spec"

            claude_spec = tool.meta["native_tools"]["claude"]
            assert claude_spec["api_type"] == "bash_20250124"
            assert claude_spec["api_name"] == "bash"

            await client.__aexit__(None, None, None)
        finally:
            with suppress(asyncio.CancelledError):
                server_task.cancel()
                await server_task

    @pytest.mark.asyncio
    async def test_computer_tool_meta_with_display_dimensions(self) -> None:
        """Test that computer tool transmits display dimensions in extra field."""
        port = _free_port()
        mcp = MCPServer(name="ComputerToolTest")

        # Create AnthropicComputerTool with custom dimensions
        computer_tool = AnthropicComputerTool(
            width=1920,
            height=1080,
        )
        mcp.add_tool(computer_tool)

        server_task = await _start_http_server(mcp, port)

        try:
            cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
            client = MCPClient({"mcpServers": cfg})
            await client.__aenter__()

            tools = await client.list_tools()
            computer_tools = [t for t in tools if "computer" in t.name]
            assert len(computer_tools) == 1

            tool = computer_tools[0]
            assert tool.meta is not None

            # Verify native_tools contains Claude spec list with display dimensions
            native_tools = tool.meta.get("native_tools", {})
            assert "claude" in native_tools

            claude_specs = native_tools["claude"]
            assert isinstance(claude_specs, list)
            assert len(claude_specs) == 2

            # First spec: computer_20251124 (Opus 4.5/4.6)
            assert claude_specs[0]["api_type"] == "computer_20251124"
            assert claude_specs[0]["role"] == "computer"
            assert claude_specs[0].get("extra", {}).get("display_width") == 1920
            assert claude_specs[0].get("extra", {}).get("display_height") == 1080

            # Second spec: computer_20250124 (catch-all)
            assert claude_specs[1]["api_type"] == "computer_20250124"
            assert claude_specs[1]["role"] == "computer"
            assert claude_specs[1].get("extra", {}).get("display_width") == 1920
            assert claude_specs[1].get("extra", {}).get("display_height") == 1080

            await client.__aexit__(None, None, None)
        finally:
            with suppress(asyncio.CancelledError):
                server_task.cancel()
                await server_task

    @pytest.mark.asyncio
    async def test_hosted_tool_meta_transmitted(self) -> None:
        """Test that hosted tools transmit hosted=True in native specs."""
        # Test hosted tool without MCP server (direct instantiation)
        google_tool = GoogleSearchTool(dynamic_threshold=0.5)

        # Check meta is properly set
        assert google_tool.meta is not None
        native_tools = google_tool.meta.get("native_tools", {})

        # Should have specs for Gemini agents
        assert "gemini" in native_tools
        gemini_spec = native_tools["gemini"]
        assert gemini_spec["api_type"] == "google_search"
        assert gemini_spec["hosted"] is True
        assert gemini_spec["extra"]["dynamic_threshold"] == 0.5


class TestNativeToolSpecAgentIntegration:
    """Test that agents correctly interpret native tool specs from MCP."""

    @pytest.mark.asyncio
    async def test_agent_categorizes_tools_from_mcp(self) -> None:
        """Test that an agent can categorize tools received from MCP server."""
        port = _free_port()
        mcp = MCPServer(name="AgentCategorizeTest")

        # Register tools
        bash_tool = BashTool()
        mcp.add_tool(bash_tool)

        @mcp.tool()
        async def generic_tool(text: str) -> str:
            """A generic tool without native specs."""
            return f"echo: {text}"

        server_task = await _start_http_server(mcp, port)

        try:
            cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
            client = MCPClient({"mcpServers": cfg})
            await client.__aenter__()

            tools = await client.list_tools()
            assert len(tools) == 2

            # Create a mock agent to test categorization
            class TestClaudeAgent(MCPAgent):
                @classmethod
                def agent_type(cls) -> AgentType:
                    return AgentType.CLAUDE

                def get_system_messages(self) -> list[Any]:
                    return []

                async def get_response(self, messages: list[Any]) -> InferenceResult:
                    return InferenceResult(content="test", done=True)

                def format_blocks(self, blocks: list[Any]) -> list[Any]:
                    return blocks

                def format_tool_results(self, results: list[Any]) -> list[Any]:
                    return results

            agent = TestClaudeAgent.create()
            # Set model to match BashTool's supported_models pattern
            agent.model = "claude-3-5-sonnet-20241022"
            agent._available_tools = list(tools)

            categorized = agent.categorize_tools()

            # BashTool should be categorized as native for Claude
            assert len(categorized.native) == 1
            assert categorized.native[0][0].name == "bash"
            assert categorized.native[0][1].api_type == "bash_20250124"

            # generic_tool should be categorized as generic
            assert len(categorized.generic) == 1
            assert categorized.generic[0].name == "generic_tool"

            await client.__aexit__(None, None, None)
        finally:
            with suppress(asyncio.CancelledError):
                server_task.cancel()
                await server_task

    @pytest.mark.asyncio
    async def test_role_exclusion_works_e2e(self) -> None:
        """Test that role-based exclusion works with mocked MCP tools."""
        from mcp import types as mcp_types

        # Create mock MCP tools as if they came from a server
        claude_computer_tool = mcp_types.Tool(
            name="anthropic_computer",
            description="Anthropic computer tool",
            inputSchema={},
            _meta={
                "native_tools": {
                    "claude": {
                        "api_type": "computer_20250124",
                        "api_name": "computer",
                        "role": "computer",
                    }
                }
            },
        )

        gemini_computer_tool = mcp_types.Tool(
            name="gemini_computer",
            description="Gemini computer tool",
            inputSchema={},
            _meta={
                "native_tools": {
                    "gemini": {
                        "api_type": "computer_use",
                        "api_name": "gemini_computer",
                        "role": "computer",
                    }
                }
            },
        )

        tools = [claude_computer_tool, gemini_computer_tool]

        # Test Claude agent - should use AnthropicComputerTool, skip Gemini one
        class TestClaudeAgent(MCPAgent):
            @classmethod
            def agent_type(cls) -> AgentType:
                return AgentType.CLAUDE

            def get_system_messages(self) -> list[Any]:
                return []

            async def get_response(self, messages: list[Any]) -> InferenceResult:
                return InferenceResult(content="test", done=True)

            def format_blocks(self, blocks: list[Any]) -> list[Any]:
                return blocks

            def format_tool_results(self, results: list[Any]) -> list[Any]:
                return results

        claude_agent = TestClaudeAgent.create()
        claude_agent._available_tools = tools
        categorized = claude_agent.categorize_tools()

        # Claude should have one native computer tool
        assert len(categorized.native) == 1
        assert "anthropic_computer" in categorized.native[0][0].name

        # Gemini computer tool should be skipped (role claimed)
        assert len(categorized.skipped) == 1
        assert "gemini_computer" in categorized.skipped[0][0].name

    @pytest.mark.asyncio
    async def test_duplicate_same_agent_computer_tools(self) -> None:
        """Test what happens when you add two computer tools for the same agent type.

        If you add two AnthropicComputerTools (or any tools with the same role for
        the same agent), the first one should be used natively and the second one
        should be skipped due to role-based exclusion.
        """
        from mcp import types as mcp_types

        # Create two Claude computer tools (simulating adding two AnthropicComputerTools)
        computer_tool_1 = mcp_types.Tool(
            name="computer_1",
            description="First computer tool (1920x1080)",
            inputSchema={},
            _meta={
                "native_tools": {
                    "claude": {
                        "api_type": "computer_20250124",
                        "api_name": "computer",
                        "role": "computer",
                        "display_width": 1920,
                        "display_height": 1080,
                    }
                }
            },
        )

        computer_tool_2 = mcp_types.Tool(
            name="computer_2",
            description="Second computer tool (1280x720)",
            inputSchema={},
            _meta={
                "native_tools": {
                    "claude": {
                        "api_type": "computer_20250124",
                        "api_name": "computer",
                        "role": "computer",
                        "display_width": 1280,
                        "display_height": 720,
                    }
                }
            },
        )

        tools = [computer_tool_1, computer_tool_2]

        class TestClaudeAgent(MCPAgent):
            @classmethod
            def agent_type(cls) -> AgentType:
                return AgentType.CLAUDE

            def get_system_messages(self) -> list[Any]:
                return []

            async def get_response(self, messages: list[Any]) -> InferenceResult:
                return InferenceResult(content="test", done=True)

            def format_blocks(self, blocks: list[Any]) -> list[Any]:
                return blocks

            def format_tool_results(self, results: list[Any]) -> list[Any]:
                return results

        claude_agent = TestClaudeAgent.create()
        claude_agent._available_tools = tools
        categorized = claude_agent.categorize_tools()

        # First computer tool should be used natively
        assert len(categorized.native) == 1
        assert categorized.native[0][0].name == "computer_1"

        # Second computer tool should be skipped (role already claimed by first)
        assert len(categorized.skipped) == 1
        assert categorized.skipped[0][0].name == "computer_2"

        # No generic tools (both have native specs)
        assert len(categorized.generic) == 0


class TestToolWithoutNativeSpecs:
    """Test backwards compatibility with tools that don't have native specs."""

    @pytest.mark.asyncio
    async def test_generic_tool_without_meta(self) -> None:
        """Test that tools without native_specs still work as generic tools."""
        port = _free_port()
        mcp = MCPServer(name="GenericToolTest")

        @mcp.tool()
        async def simple_tool(text: str) -> str:
            """A simple tool with no native specs."""
            return f"result: {text}"

        server_task = await _start_http_server(mcp, port)

        try:
            cfg = {"srv": {"url": f"http://127.0.0.1:{port}/mcp"}}
            client = MCPClient({"mcpServers": cfg})
            await client.__aenter__()

            tools = await client.list_tools()
            assert len(tools) == 1

            tool = tools[0]
            assert tool.name == "simple_tool"

            # meta might be None or empty - both are valid
            native_tools = (tool.meta or {}).get("native_tools", {})
            assert native_tools == {}

            # Test that agent handles this correctly
            class TestAgent(MCPAgent):
                @classmethod
                def agent_type(cls) -> AgentType:
                    return AgentType.CLAUDE

                def get_system_messages(self) -> list[Any]:
                    return []

                async def get_response(self, messages: list[Any]) -> InferenceResult:
                    return InferenceResult(content="test", done=True)

                def format_blocks(self, blocks: list[Any]) -> list[Any]:
                    return blocks

                def format_tool_results(self, results: list[Any]) -> list[Any]:
                    return results

            agent = TestAgent.create()
            agent._available_tools = list(tools)
            categorized = agent.categorize_tools()

            # Tool should be categorized as generic
            assert len(categorized.native) == 0
            assert len(categorized.hosted) == 0
            assert len(categorized.generic) == 1
            assert categorized.generic[0].name == "simple_tool"

            await client.__aexit__(None, None, None)
        finally:
            with suppress(asyncio.CancelledError):
                server_task.cancel()
                await server_task


class TestLegacyNameFallback:
    """Test that old environments without native_tools metadata work via name-based fallback."""

    @pytest.fixture
    def mock_anthropic(self) -> Any:
        """Create a mock Anthropic client."""
        from unittest.mock import MagicMock

        return MagicMock(spec=["messages", "beta"])

    @pytest.fixture
    def mock_openai(self) -> Any:
        """Create a mock OpenAI client."""
        from unittest.mock import MagicMock

        return MagicMock(spec=["responses", "chat"])

    @pytest.fixture
    def mock_gemini(self) -> Any:
        """Create a mock Gemini client."""
        from unittest.mock import MagicMock

        return MagicMock()

    def test_claude_legacy_computer_fallback(self, mock_anthropic: Any) -> None:
        """Test Claude agent detects anthropic_computer by name without metadata."""
        from mcp import types as mcp_types

        from hud.agents.claude import ClaudeAgent

        # Create a tool with NO native_tools metadata - just a name
        legacy_tool = mcp_types.Tool(
            name="anthropic_computer",
            description="Old-style computer tool without native_tools metadata",
            inputSchema={"type": "object", "properties": {}},
            # Note: NO _meta field at all!
        )

        agent = ClaudeAgent.create(model_client=mock_anthropic, validate_api_key=False)
        agent._available_tools = [legacy_tool]

        # The legacy fallback should detect this as a computer tool
        spec = agent.resolve_native_spec(legacy_tool)
        assert spec is not None, "Legacy fallback should detect anthropic_computer"
        assert spec.api_type == "computer_20250124"
        assert spec.role == "computer"

        # Categorize should work
        categorized = agent.categorize_tools()
        assert len(categorized.native) == 1
        assert categorized.native[0][0].name == "anthropic_computer"
        assert categorized.native[0][1].api_type == "computer_20250124"

    def test_claude_legacy_bash_fallback(self, mock_anthropic: Any) -> None:
        """Test Claude agent detects bash by name without metadata."""
        from mcp import types as mcp_types

        from hud.agents.claude import ClaudeAgent

        legacy_tool = mcp_types.Tool(
            name="bash",
            description="Old-style bash tool",
            inputSchema={"type": "object", "properties": {}},
        )

        agent = ClaudeAgent.create(model_client=mock_anthropic, validate_api_key=False)
        agent._available_tools = [legacy_tool]

        spec = agent.resolve_native_spec(legacy_tool)
        assert spec is not None, "Legacy fallback should detect bash"
        assert spec.api_type == "bash_20250124"

    def test_claude_legacy_editor_fallback(self, mock_anthropic: Any) -> None:
        """Test Claude agent detects str_replace_based_edit_tool by name."""
        from mcp import types as mcp_types

        from hud.agents.claude import ClaudeAgent

        legacy_tool = mcp_types.Tool(
            name="str_replace_based_edit_tool",
            description="Old-style editor tool",
            inputSchema={"type": "object", "properties": {}},
        )

        agent = ClaudeAgent.create(model_client=mock_anthropic, validate_api_key=False)
        agent._available_tools = [legacy_tool]

        spec = agent.resolve_native_spec(legacy_tool)
        assert spec is not None, "Legacy fallback should detect editor"
        assert spec.api_type == "text_editor_20250728"

    def test_gemini_legacy_computer_fallback(self, mock_gemini: Any) -> None:
        """Test Gemini agent detects gemini_computer by name without metadata."""
        from mcp import types as mcp_types

        from hud.agents.gemini import GeminiAgent

        legacy_tool = mcp_types.Tool(
            name="gemini_computer",
            description="Old-style Gemini computer tool",
            inputSchema={"type": "object", "properties": {}},
        )

        agent = GeminiAgent.create(model_client=mock_gemini, validate_api_key=False)
        agent._available_tools = [legacy_tool]

        spec = agent.resolve_native_spec(legacy_tool)
        assert spec is not None, "Legacy fallback should detect gemini_computer"
        assert spec.api_type == "computer_use"
        assert spec.role == "computer"

    def test_gemini_cua_legacy_computer_fallback(self, mock_gemini: Any) -> None:
        """Test GeminiCUAAgent detects gemini_computer by name without metadata."""
        from mcp import types as mcp_types

        from hud.agents.gemini_cua import GeminiCUAAgent

        legacy_tool = mcp_types.Tool(
            name="gemini_computer",
            description="Old-style Gemini CUA computer tool",
            inputSchema={"type": "object", "properties": {}},
        )

        agent = GeminiCUAAgent.create(model_client=mock_gemini, validate_api_key=False)
        agent._available_tools = [legacy_tool]

        spec = agent.resolve_native_spec(legacy_tool)
        assert spec is not None, "Legacy fallback should detect gemini_computer for CUA"
        assert spec.api_type == "computer_use"
        assert spec.role == "computer"

    def test_openai_legacy_shell_fallback(self, mock_openai: Any) -> None:
        """Test OpenAI agent detects shell by name without metadata."""
        from mcp import types as mcp_types

        from hud.agents.openai import OpenAIAgent

        legacy_tool = mcp_types.Tool(
            name="shell",
            description="Old-style shell tool",
            inputSchema={"type": "object", "properties": {}},
        )

        agent = OpenAIAgent.create(model_client=mock_openai, validate_api_key=False)
        agent._available_tools = [legacy_tool]

        spec = agent.resolve_native_spec(legacy_tool)
        assert spec is not None, "Legacy fallback should detect shell"
        assert spec.api_type == "shell"

    def test_openai_legacy_apply_patch_fallback(self, mock_openai: Any) -> None:
        """Test OpenAI agent detects apply_patch by name without metadata."""
        from mcp import types as mcp_types

        from hud.agents.openai import OpenAIAgent

        legacy_tool = mcp_types.Tool(
            name="apply_patch",
            description="Old-style apply_patch tool",
            inputSchema={"type": "object", "properties": {}},
        )

        agent = OpenAIAgent.create(model_client=mock_openai, validate_api_key=False)
        agent._available_tools = [legacy_tool]

        spec = agent.resolve_native_spec(legacy_tool)
        assert spec is not None, "Legacy fallback should detect apply_patch"
        assert spec.api_type == "apply_patch"

    def test_metadata_takes_precedence_over_legacy(self, mock_anthropic: Any) -> None:
        """Test that explicit native_tools metadata takes precedence over name matching."""
        from mcp import types as mcp_types

        from hud.agents.claude import ClaudeAgent

        # Tool named "computer" but with custom metadata
        tool_with_metadata = mcp_types.Tool(
            name="computer",
            description="Computer tool with explicit metadata",
            inputSchema={"type": "object", "properties": {}},
            _meta={
                "native_tools": {
                    "claude": {
                        "api_type": "computer_20250124",
                        "api_name": "computer",
                        "role": "computer",
                        "display_width": 1920,  # Custom dimensions
                        "display_height": 1080,
                    }
                }
            },
        )

        agent = ClaudeAgent.create(model_client=mock_anthropic, validate_api_key=False)
        agent._available_tools = [tool_with_metadata]

        spec = agent.resolve_native_spec(tool_with_metadata)
        assert spec is not None
        assert spec.api_type == "computer_20250124"
        # Metadata should be used, including custom dimensions
        assert spec.extra.get("display_width") == 1920
        assert spec.extra.get("display_height") == 1080


class TestBackwardsCompatibility:
    """Test backwards compatibility with old-style tools and settings fallbacks."""

    @pytest.fixture
    def mock_anthropic(self) -> Any:
        """Create a mock Anthropic client."""
        from unittest.mock import MagicMock

        return MagicMock(spec=["messages", "beta"])

    def test_computer_tool_without_display_dims_uses_fallback(self, mock_anthropic: Any) -> None:
        """Test that a native spec without display dimensions falls back to settings."""
        import warnings

        from mcp import types as mcp_types

        from hud.agents.claude import ClaudeAgent
        from hud.tools.computer.settings import computer_settings

        # Create a mock tool with native spec but NO display dimensions in extra
        computer_tool = mcp_types.Tool(
            name="computer",
            description="Old-style computer tool without display dims",
            inputSchema={},
            _meta={
                "native_tools": {
                    "claude": {
                        "api_type": "computer_20250124",
                        "api_name": "computer",
                        "role": "computer",
                        # Note: NO display_width or display_height in extra
                    }
                }
            },
        )

        # Create agent and get native spec
        agent = ClaudeAgent.create(model_client=mock_anthropic, validate_api_key=False)
        agent._available_tools = [computer_tool]
        spec = agent.resolve_native_spec(computer_tool)

        assert spec is not None
        assert spec.api_type == "computer_20250124"

        # The spec.extra should be empty (no display dimensions)
        assert spec.extra.get("display_width") is None
        assert spec.extra.get("display_height") is None

        # When building native tool, it should fall back to computer_settings
        # and emit a deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            claude_tool = agent._build_native_tool(computer_tool, spec)

            # Should have emitted deprecation warning
            deprecation_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "display dimensions" in str(deprecation_warnings[0].message).lower()
            assert "v0.6.0" in str(deprecation_warnings[0].message)

        # Tool should still work with fallback dimensions
        # Cast to Any for TypedDict union access
        tool_dict = cast("dict[str, Any]", claude_tool)
        assert tool_dict["display_width_px"] == computer_settings.ANTHROPIC_COMPUTER_WIDTH
        assert tool_dict["display_height_px"] == computer_settings.ANTHROPIC_COMPUTER_HEIGHT

    def test_new_style_tool_with_display_dims_no_warning(self, mock_anthropic: Any) -> None:
        """Test that a new-style tool with display dimensions doesn't emit warning."""
        import warnings

        from mcp import types as mcp_types

        from hud.agents.claude import ClaudeAgent

        # Create a tool with display dimensions at the top level (non-standard fields go to extra)
        # This is how NativeToolSpec.model_dump() serializes it
        computer_tool = mcp_types.Tool(
            name="computer",
            description="New-style computer tool with display dims",
            inputSchema={},
            _meta={
                "native_tools": {
                    "claude": {
                        "api_type": "computer_20250124",
                        "api_name": "computer",
                        "role": "computer",
                        # display_width/height are non-standard fields,
                        # so resolve_native_spec puts them in extra
                        "display_width": 1920,
                        "display_height": 1080,
                    }
                }
            },
        )

        agent = ClaudeAgent.create(model_client=mock_anthropic, validate_api_key=False)
        agent._available_tools = [computer_tool]
        spec = agent.resolve_native_spec(computer_tool)

        assert spec is not None
        assert spec.extra.get("display_width") == 1920
        assert spec.extra.get("display_height") == 1080

        # Should NOT emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            claude_tool = agent._build_native_tool(computer_tool, spec)

            deprecation_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0

        # Tool should use provided dimensions
        # Cast to Any for TypedDict union access
        tool_dict = cast("dict[str, Any]", claude_tool)
        assert tool_dict["display_width_px"] == 1920
        assert tool_dict["display_height_px"] == 1080


class TestToolNativeSpecs:
    """Tests for native_specs on tool classes."""

    def test_shell_tool_has_openai_native_spec(self) -> None:
        """Test ShellTool has native_specs for OpenAI."""
        from hud.tools.coding import ShellTool
        from hud.types import AgentType

        assert hasattr(ShellTool, "native_specs")
        assert AgentType.OPENAI in ShellTool.native_specs
        spec = ShellTool.native_specs[AgentType.OPENAI]
        assert isinstance(spec, NativeToolSpec)
        assert spec.api_type == "shell"
        assert spec.api_name == "shell"

    def test_apply_patch_tool_has_openai_native_spec(self) -> None:
        """Test ApplyPatchTool has native_specs for OpenAI."""
        from hud.tools.coding import ApplyPatchTool
        from hud.types import AgentType

        assert hasattr(ApplyPatchTool, "native_specs")
        assert AgentType.OPENAI in ApplyPatchTool.native_specs
        spec = ApplyPatchTool.native_specs[AgentType.OPENAI]
        assert isinstance(spec, NativeToolSpec)
        assert spec.api_type == "apply_patch"
        assert spec.api_name == "apply_patch"

    def test_bash_tool_has_claude_native_spec(self) -> None:
        """Test BashTool has native_specs for Claude."""
        from hud.tools.coding import BashTool
        from hud.types import AgentType

        assert hasattr(BashTool, "native_specs")
        assert AgentType.CLAUDE in BashTool.native_specs
        spec = BashTool.native_specs[AgentType.CLAUDE]
        assert isinstance(spec, NativeToolSpec)
        assert spec.api_type == "bash_20250124"
        assert spec.api_name == "bash"

    def test_edit_tool_has_claude_native_spec(self) -> None:
        """Test EditTool has native_specs for Claude."""
        from hud.tools.coding import EditTool
        from hud.types import AgentType

        assert hasattr(EditTool, "native_specs")
        assert AgentType.CLAUDE in EditTool.native_specs
        spec = EditTool.native_specs[AgentType.CLAUDE]
        assert isinstance(spec, NativeToolSpec)
        assert spec.api_type == "text_editor_20250728"
        assert spec.api_name == "str_replace_based_edit_tool"

    def test_shell_tools_have_mutual_exclusion_role(self) -> None:
        """Test BashTool and ShellTool both have role='shell' for mutual exclusion."""
        from hud.tools.coding import BashTool, ShellTool
        from hud.types import AgentType

        bash_spec = BashTool.native_specs[AgentType.CLAUDE]
        shell_spec = ShellTool.native_specs[AgentType.OPENAI]
        assert isinstance(bash_spec, NativeToolSpec)
        assert isinstance(shell_spec, NativeToolSpec)

        assert bash_spec.role == "shell"
        assert shell_spec.role == "shell"

    def test_editor_tools_have_mutual_exclusion_role(self) -> None:
        """Test EditTool and ApplyPatchTool both have role='editor' for mutual exclusion."""
        from hud.tools.coding import ApplyPatchTool, EditTool
        from hud.types import AgentType

        edit_spec = EditTool.native_specs[AgentType.CLAUDE]
        apply_patch_spec = ApplyPatchTool.native_specs[AgentType.OPENAI]
        assert isinstance(edit_spec, NativeToolSpec)
        assert isinstance(apply_patch_spec, NativeToolSpec)

        assert edit_spec.role == "editor"
        assert apply_patch_spec.role == "editor"

    def test_gemini_tools_have_role_but_not_native(self) -> None:
        """Test GeminiShellTool and GeminiEditTool have roles but no native API."""
        from hud.tools.coding import GeminiEditTool, GeminiShellTool
        from hud.types import AgentType

        shell_spec = GeminiShellTool.native_specs[AgentType.GEMINI]
        edit_spec = GeminiEditTool.native_specs[AgentType.GEMINI]
        assert isinstance(shell_spec, NativeToolSpec)
        assert isinstance(edit_spec, NativeToolSpec)

        # Should have roles for mutual exclusion
        assert shell_spec.role == "shell"
        assert edit_spec.role == "editor"

        # But should NOT be native (no api_type means standard function calling)
        assert shell_spec.api_type is None
        assert edit_spec.api_type is None
        assert shell_spec.is_native is False
        assert edit_spec.is_native is False
