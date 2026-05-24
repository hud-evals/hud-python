"""Agent-side Claude native memory tool backed by an environment tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from .base import ClaudeTool, ClaudeToolSpec

if TYPE_CHECKING:
    from anthropic.types.beta import BetaToolUnionParam


CLAUDE_MEMORY_SPEC = ClaudeToolSpec(
    api_type="memory_20250818",
    api_name="memory",
    supported_models=(
        "claude-opus-4-7*",
        "claude-opus-4-6*",
        "claude-sonnet-4-6*",
        "claude-haiku-4-5*",
    ),
)


class ClaudeMemoryTool(ClaudeTool):
    """Claude memory provider tool backed by an environment memory tool."""

    name = "memory"
    capability = "memory"

    @classmethod
    def default_spec(cls, model: str) -> ClaudeToolSpec | None:
        if CLAUDE_MEMORY_SPEC.supports_model(model):
            return CLAUDE_MEMORY_SPEC
        return None

    def __init__(self, *, env_tool_name: str, spec: ClaudeToolSpec) -> None:
        del spec
        super().__init__(env_tool_name=env_tool_name, spec=CLAUDE_MEMORY_SPEC)

    def to_params(self) -> BetaToolUnionParam:
        return cast(
            "BetaToolUnionParam",
            {
                "type": "memory_20250818",
                "name": self.name,
            },
        )
