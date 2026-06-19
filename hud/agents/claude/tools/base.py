"""Claude-specific tool spec."""

from __future__ import annotations

from dataclasses import dataclass

from hud.agents.tools.base import AgentToolSpec


@dataclass(frozen=True)
class ClaudeToolSpec(AgentToolSpec):
    """Claude tool spec — adds the optional Anthropic beta flag."""

    beta: str | None = None


__all__ = ["ClaudeToolSpec"]
