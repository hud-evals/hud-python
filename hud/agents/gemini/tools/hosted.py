"""Gemini hosted tools configured by the Gemini harness."""

from __future__ import annotations

from dataclasses import dataclass

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
        if self.dynamic_threshold is not None:
            raise ValueError("dynamic_threshold is not supported by Gemini Google Search.")
        return genai_types.Tool(google_search=genai_types.GoogleSearch())


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
