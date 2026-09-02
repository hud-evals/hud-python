"""Claude-specific tool spec."""

from __future__ import annotations

from dataclasses import dataclass

from hud.agents.tools.base import AgentToolSpec


@dataclass(frozen=True)
class ClaudeToolSpec(AgentToolSpec):
    """Claude tool spec — adds the optional Anthropic beta flag."""

    beta: str | None = None


def is_anthropic_model(model: str | None) -> bool:
    """Whether the model accepts Anthropic server-tool shorthands (``bash_20250124`` etc.).

    Non-Anthropic models served over Anthropic-compatible endpoints (e.g. OpenRouter)
    reject those shorthands, so tools fall back to plain ``input_schema`` definitions.
    """
    return bool(model) and "claude" in model.lower()


__all__ = ["ClaudeToolSpec", "is_anthropic_model"]
