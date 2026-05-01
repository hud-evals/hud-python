"""Gemini hosted tools configured by the Gemini harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google.genai import types as genai_types

from hud.agents.tools import HostedTool


@dataclass(frozen=True, kw_only=True)
class GeminiHostedTool(HostedTool[genai_types.Tool]):
    """Gemini-hosted tool configured by the Gemini harness."""


@dataclass(frozen=True, kw_only=True)
class GeminiGoogleSearchTool(GeminiHostedTool):
    """Gemini Google Search."""

    dynamic_threshold: float | None = None

    def to_params(self) -> genai_types.Tool:
        kwargs: dict[str, Any] = {}
        if self.dynamic_threshold is not None:
            kwargs["dynamic_threshold"] = self.dynamic_threshold
        try:
            google_search = genai_types.GoogleSearch(**kwargs)
        except Exception:
            google_search = genai_types.GoogleSearch()
        return genai_types.Tool(google_search=google_search)


@dataclass(frozen=True, kw_only=True)
class GeminiUrlContextTool(GeminiHostedTool):
    """Gemini URL context."""

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(url_context=genai_types.UrlContext())


@dataclass(frozen=True, kw_only=True)
class GeminiCodeExecutionTool(GeminiHostedTool):
    """Gemini code execution."""

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(code_execution=genai_types.ToolCodeExecution())


__all__ = [
    "GeminiCodeExecutionTool",
    "GeminiGoogleSearchTool",
    "GeminiHostedTool",
    "GeminiUrlContextTool",
]
