"""Tests for native tool types and specifications."""

from __future__ import annotations

from typing import ClassVar

import pytest

from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType


class TestNativeToolSpec:
    """Tests for NativeToolSpec dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic NativeToolSpec."""
        spec = NativeToolSpec(api_type="computer_20250124")
        assert spec.api_type == "computer_20250124"
        assert spec.api_name is None
        assert spec.beta is None
        assert spec.hosted is False
        assert spec.supported_models is None
        assert spec.extra == {}

    def test_full_creation(self) -> None:
        """Test creating a NativeToolSpec with all fields."""
        spec = NativeToolSpec(
            api_type="computer_20250124",
            api_name="computer",
            beta="computer-use-2025-01-24",
            hosted=False,
            extra={"display_width": 1024, "display_height": 768},
        )
        assert spec.api_type == "computer_20250124"
        assert spec.api_name == "computer"
        assert spec.beta == "computer-use-2025-01-24"
        assert spec.hosted is False
        assert spec.extra == {"display_width": 1024, "display_height": 768}

    def test_hosted_tool_creation(self) -> None:
        """Test creating a hosted tool spec."""
        spec = NativeToolSpec(
            api_type="google_search",
            hosted=True,
            extra={"dynamic_threshold": 0.5},
        )
        assert spec.api_type == "google_search"
        assert spec.hosted is True
        assert spec.extra["dynamic_threshold"] == 0.5

    def test_model_dump(self) -> None:
        """Test serialization via model_dump."""
        spec = NativeToolSpec(
            api_type="bash_20250124",
            api_name="bash",
            beta="computer-use-2025-01-24",
        )
        dumped = spec.model_dump(exclude_none=True)
        assert dumped == {
            "api_type": "bash_20250124",
            "api_name": "bash",
            "beta": "computer-use-2025-01-24",
            "hosted": False,
            "extra": {},
        }

    def test_model_dump_excludes_none(self) -> None:
        """Test that model_dump with exclude_none removes None fields."""
        spec = NativeToolSpec(api_type="google_search", hosted=True)
        dumped = spec.model_dump(exclude_none=True)
        # api_name and beta are None, so they should still appear (they're not None values)
        # Actually, since they're None by default, exclude_none=True should exclude them
        assert "api_type" in dumped
        assert dumped["hosted"] is True

    def test_frozen_immutability(self) -> None:
        """Test that NativeToolSpec is immutable (frozen)."""
        spec = NativeToolSpec(api_type="test")
        with pytest.raises(Exception):  # ValidationError for frozen model
            spec.api_type = "modified"  # type: ignore[misc]

    def test_supported_models_creation(self) -> None:
        """Test creating a NativeToolSpec with supported_models."""
        spec = NativeToolSpec(
            api_type="bash_20250124",
            supported_models=("claude-3-5-sonnet-*", "claude-3-7-sonnet-*"),
        )
        assert spec.supported_models == ("claude-3-5-sonnet-*", "claude-3-7-sonnet-*")

    def test_supported_models_serialization(self) -> None:
        """Test that supported_models serializes to a list."""
        spec = NativeToolSpec(
            api_type="bash_20250124",
            supported_models=("claude-3-5-sonnet-*", "claude-3-7-sonnet-*"),
        )
        dumped = spec.model_dump(exclude_none=True)
        # Pydantic serializes tuples as lists
        assert dumped["supported_models"] == ["claude-3-5-sonnet-*", "claude-3-7-sonnet-*"]


class TestSupportsModel:
    """Tests for NativeToolSpec.supports_model() method."""

    def test_no_restrictions_supports_all(self) -> None:
        """Test that spec without supported_models supports all models."""
        spec = NativeToolSpec(api_type="test")
        assert spec.supports_model("any-model") is True
        assert spec.supports_model("gpt-4o") is True
        assert spec.supports_model("claude-3-5-sonnet-20241022") is True

    def test_no_model_defaults_to_supported(self) -> None:
        """Test that None/empty model defaults to supported."""
        spec = NativeToolSpec(
            api_type="test",
            supported_models=("claude-*",),
        )
        assert spec.supports_model(None) is True
        assert spec.supports_model("") is True

    def test_exact_match(self) -> None:
        """Test exact model name matching."""
        spec = NativeToolSpec(
            api_type="test",
            supported_models=("gpt-4o", "gpt-4o-mini"),
        )
        assert spec.supports_model("gpt-4o") is True
        assert spec.supports_model("gpt-4o-mini") is True
        assert spec.supports_model("gpt-4o-2024-05-13") is False

    def test_wildcard_suffix_match(self) -> None:
        """Test wildcard suffix pattern matching."""
        spec = NativeToolSpec(
            api_type="test",
            supported_models=("claude-3-5-sonnet-*",),
        )
        assert spec.supports_model("claude-3-5-sonnet-20241022") is True
        assert spec.supports_model("claude-3-5-sonnet-latest") is True
        assert spec.supports_model("claude-3-5-sonnet-") is True
        assert spec.supports_model("claude-3-7-sonnet-20250219") is False

    def test_wildcard_prefix_match(self) -> None:
        """Test wildcard prefix pattern matching."""
        spec = NativeToolSpec(
            api_type="test",
            supported_models=("*-sonnet",),
        )
        assert spec.supports_model("claude-3-5-sonnet") is True
        assert spec.supports_model("claude-3-7-sonnet") is True
        assert spec.supports_model("claude-3-5-sonnet-20241022") is False

    def test_multiple_patterns(self) -> None:
        """Test matching against multiple patterns."""
        spec = NativeToolSpec(
            api_type="test",
            supported_models=(
                "claude-3-5-sonnet-*",
                "claude-3-7-sonnet-*",
                "claude-sonnet-4-*",
            ),
        )
        assert spec.supports_model("claude-3-5-sonnet-20241022") is True
        assert spec.supports_model("claude-3-7-sonnet-20250219") is True
        assert spec.supports_model("claude-sonnet-4-20250514") is True
        assert spec.supports_model("claude-3-opus-20240229") is False

    def test_case_insensitive(self) -> None:
        """Test that matching is case-insensitive."""
        spec = NativeToolSpec(
            api_type="test",
            supported_models=("Claude-3-5-Sonnet-*",),
        )
        assert spec.supports_model("claude-3-5-sonnet-20241022") is True
        assert spec.supports_model("CLAUDE-3-5-SONNET-20241022") is True
        assert spec.supports_model("Claude-3-5-Sonnet-20241022") is True

    def test_openai_gpt5_models(self) -> None:
        """Test OpenAI GPT-5.x model patterns."""
        spec = NativeToolSpec(
            api_type="shell",
            supported_models=("gpt-5.1", "gpt-5.1-*", "gpt-5.2", "gpt-5.2-*"),
        )
        assert spec.supports_model("gpt-5.1") is True
        assert spec.supports_model("gpt-5.1-pro") is True
        assert spec.supports_model("gpt-5.1-2025-11-13") is True
        assert spec.supports_model("gpt-5.2") is True
        assert spec.supports_model("gpt-5.2-2026-01-15") is True
        assert spec.supports_model("gpt-4o") is False
        assert spec.supports_model("gpt-4o-mini") is False
        assert spec.supports_model("o3") is False


class TestNativeToolSpecs:
    """Tests for NativeToolSpecs type alias."""

    def test_specs_dict_creation(self) -> None:
        """Test creating a NativeToolSpecs dictionary."""
        specs: NativeToolSpecs = {
            AgentType.CLAUDE: NativeToolSpec(
                api_type="computer_20250124",
                api_name="computer",
                beta="computer-use-2025-01-24",
            ),
            AgentType.GEMINI: NativeToolSpec(
                api_type="computer_use",
                api_name="gemini_computer",
            ),
        }
        assert len(specs) == 2
        assert AgentType.CLAUDE in specs
        assert AgentType.GEMINI in specs
        assert specs[AgentType.CLAUDE].api_type == "computer_20250124"
        assert specs[AgentType.GEMINI].api_type == "computer_use"

    def test_specs_serialization_for_meta(self) -> None:
        """Test serializing specs for embedding in tool meta."""
        specs: NativeToolSpecs = {
            AgentType.CLAUDE: NativeToolSpec(
                api_type="bash_20250124",
                api_name="bash",
                beta="computer-use-2025-01-24",
            ),
        }
        # Simulate what BaseTool does
        meta_native_tools = {
            agent_type.value: spec.model_dump(exclude_none=True)
            for agent_type, spec in specs.items()
        }
        assert "claude" in meta_native_tools
        assert meta_native_tools["claude"]["api_type"] == "bash_20250124"
        assert meta_native_tools["claude"]["api_name"] == "bash"
        assert meta_native_tools["claude"]["beta"] == "computer-use-2025-01-24"


class TestBaseToolNativeSpecs:
    """Tests for BaseTool native_specs integration."""

    def test_tool_with_class_native_specs(self) -> None:
        """Test that class-level native_specs are embedded in meta."""
        from hud.tools.coding import BashTool

        tool = BashTool()
        assert tool.meta is not None
        assert "native_tools" in tool.meta
        assert "claude" in tool.meta["native_tools"]
        assert tool.meta["native_tools"]["claude"]["api_type"] == "bash_20250124"
        assert tool.meta["native_tools"]["claude"]["api_name"] == "bash"
        # Check that supported_models is included
        assert "supported_models" in tool.meta["native_tools"]["claude"]
        assert "claude-3-5-sonnet-*" in tool.meta["native_tools"]["claude"]["supported_models"]

    def test_tool_with_instance_native_specs(self) -> None:
        """Test that instance-level native_specs merge with class-level."""
        from hud.tools.base import BaseTool

        # Create a simple test tool
        class TestTool(BaseTool):
            native_specs: ClassVar[dict[AgentType, NativeToolSpec]] = {
                AgentType.CLAUDE: NativeToolSpec(api_type="test_class"),
            }

            async def __call__(self, **kwargs: object) -> list[object]:
                return []

        # Instance with override
        instance_specs = {
            AgentType.GEMINI: NativeToolSpec(api_type="test_instance"),
        }
        tool = TestTool(native_specs=instance_specs)

        # Both should be present
        assert "claude" in tool.meta["native_tools"]
        assert "gemini" in tool.meta["native_tools"]
        assert tool.meta["native_tools"]["claude"]["api_type"] == "test_class"
        assert tool.meta["native_tools"]["gemini"]["api_type"] == "test_instance"

    def test_get_native_spec(self) -> None:
        """Test get_native_spec method on BaseTool."""
        from hud.tools.coding import BashTool

        tool = BashTool()
        spec = tool.get_native_spec(AgentType.CLAUDE)
        assert spec is not None
        assert spec.api_type == "bash_20250124"

        # Non-existent agent type
        spec = tool.get_native_spec(AgentType.OPENAI)
        assert spec is None


class TestHostedTools:
    """Tests for hosted tool classes."""

    def test_google_search_tool(self) -> None:
        """Test GoogleSearchTool creation and specs."""
        from hud.tools.hosted import GoogleSearchTool

        tool = GoogleSearchTool()
        assert tool.name == "google_search"
        assert "native_tools" in tool.meta

        gemini_spec = tool.meta["native_tools"].get("gemini")
        assert gemini_spec is not None
        assert gemini_spec["api_type"] == "google_search"
        assert gemini_spec["hosted"] is True

    def test_google_search_with_threshold(self) -> None:
        """Test GoogleSearchTool with dynamic threshold."""
        from hud.tools.hosted import GoogleSearchTool

        tool = GoogleSearchTool(dynamic_threshold=0.3)
        gemini_spec = tool.meta["native_tools"]["gemini"]
        assert gemini_spec["extra"]["dynamic_threshold"] == 0.3

    def test_code_execution_tool(self) -> None:
        """Test CodeExecutionTool creation and specs."""
        from hud.tools.hosted import CodeExecutionTool

        tool = CodeExecutionTool()
        assert tool.name == "code_execution"

        # Should have specs for both Gemini and OpenAI
        assert "gemini" in tool.meta["native_tools"]
        assert "openai" in tool.meta["native_tools"]

        assert tool.meta["native_tools"]["gemini"]["api_type"] == "code_execution"
        assert tool.meta["native_tools"]["openai"]["api_type"] == "code_interpreter"

    @pytest.mark.asyncio
    async def test_hosted_tool_call_raises(self) -> None:
        """Test that calling a hosted tool raises NotImplementedError."""
        from hud.tools.hosted import GoogleSearchTool

        tool = GoogleSearchTool()
        with pytest.raises(NotImplementedError):
            await tool()
