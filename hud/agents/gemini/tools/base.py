"""Base Gemini agent-owned tool types."""

from __future__ import annotations

from typing import Any, ClassVar

from google.genai import types as genai_types

from hud.agents.tools import AgentTool, AgentToolSpec, CallTool, call_tool

GeminiToolSpec = AgentToolSpec


class GeminiTool(AgentTool[Any]):
    """Gemini provider tool backed by an environment tool."""


class GeminiFunctionTool(GeminiTool):
    """Gemini function declaration backed by an environment tool."""

    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    def to_params(self) -> genai_types.Tool:
        return genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name=self.name,
                    description=self.description,
                    parameters_json_schema=self.parameters,
                )
            ]
        )


__all__ = ["CallTool", "GeminiFunctionTool", "GeminiTool", "GeminiToolSpec", "call_tool"]
