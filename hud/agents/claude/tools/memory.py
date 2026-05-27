"""Claude native memory tool — provider-hosted, no env capability needed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from hud.agents.tools.hosted import HostedTool

if TYPE_CHECKING:
    from anthropic.types.beta import BetaToolUnionParam


@dataclass(frozen=True, kw_only=True)
class ClaudeMemoryTool(HostedTool["BetaToolUnionParam"]):
    """Claude's built-in memory tool (``memory_20250818``).

    This is provider-hosted — Anthropic manages the storage server-side.
    Add it to ``hosted_tools`` in the agent config.
    """

    supported_models: tuple[str, ...] | None = (
        "claude-opus-4-7*",
        "claude-opus-4-6*",
        "claude-sonnet-4-6*",
        "claude-haiku-4-5*",
    )

    def to_params(self) -> BetaToolUnionParam:
        return cast(
            "BetaToolUnionParam",
            {"type": "memory_20250818", "name": "memory"},
        )


__all__ = ["ClaudeMemoryTool"]
