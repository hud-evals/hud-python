"""Gemini-specific tool spec."""

from __future__ import annotations

from typing import Any

from google.genai import types as genai_types

from hud.agents.tools.base import AgentToolSpec

GeminiToolSpec = AgentToolSpec


def declaration(name: str, description: str, parameters: dict[str, Any]) -> genai_types.Tool:
    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name=name,
                description=description,
                parameters_json_schema=parameters,
            ),
        ],
    )


def required_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value


__all__ = ["GeminiToolSpec", "declaration", "required_str"]
