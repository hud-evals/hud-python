"""Provider-executed tool search for large tool sets."""

from __future__ import annotations

from typing import ClassVar

from hud.tools.hosted.base import HostedTool
from hud.tools.native_types import NativeToolSpec, NativeToolSpecs
from hud.types import AgentType


class ToolSearchTool(HostedTool):
    """Provider-executed tool search that indexes function tools server-side.

    When enabled and the number of function tools exceeds `threshold`, the API
    marks all function tools with ``defer_loading: True`` and adds a search
    entry so the model can discover relevant tools on demand.

    Supported by OpenAI (tool_search) and Claude (tool_search_tool_bm25).
    """

    _openai_models: ClassVar[tuple[str, ...]] = (
        "gpt-5.4",
        "gpt-5.4-*",
    )
    _claude_models: ClassVar[tuple[str, ...]] = (
        "claude-sonnet-4-5*",
        "claude-sonnet-4-6*",
        "claude-opus-4-5*",
        "claude-opus-4-6*",
    )

    native_specs: ClassVar[NativeToolSpecs] = {
        AgentType.OPENAI: NativeToolSpec(
            api_type="tool_search",
            hosted=True,
            supported_models=(
                "gpt-5.4",
                "gpt-5.4-*",
            ),
        ),
        AgentType.CLAUDE: NativeToolSpec(
            api_type="tool_search_tool_bm25_20251119",
            api_name="tool_search_tool_bm25",
            hosted=True,
            supported_models=(
                "claude-sonnet-4-5*",
                "claude-sonnet-4-6*",
                "claude-opus-4-5*",
                "claude-opus-4-6*",
            ),
        ),
    }

    def __init__(self, threshold: int = 10) -> None:
        """Initialize ToolSearchTool.

        Args:
            threshold: Minimum number of function tools before tool search activates.
                       Below this count, the tool is a no-op.
        """
        instance_specs: NativeToolSpecs = {
            AgentType.OPENAI: NativeToolSpec(
                api_type="tool_search",
                hosted=True,
                extra={"threshold": threshold},
                supported_models=self._openai_models,
            ),
            AgentType.CLAUDE: NativeToolSpec(
                api_type="tool_search_tool_bm25_20251119",
                api_name="tool_search_tool_bm25",
                hosted=True,
                extra={"threshold": threshold},
                supported_models=self._claude_models,
            ),
        }
        super().__init__(
            name="tool_search",
            title="Tool Search",
            description="Server-side tool search for large tool sets",
            native_specs=instance_specs,
        )
