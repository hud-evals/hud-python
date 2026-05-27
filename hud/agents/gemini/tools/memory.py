"""Gemini memory tool — backed by SSHClient (writes to /memories/)."""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

from google.genai import types as genai_types

from hud.agents.tools import SSHTool
from hud.types import MCPToolResult

from .base import GeminiToolSpec
from .coding import _decl

GEMINI_MEMORY_SPEC = GeminiToolSpec(api_type="save_memory", api_name="save_memory")


class GeminiMemoryTool(SSHTool):
    name = "save_memory"
    description: ClassVar[str] = "Saves a specific fact to long-term memory."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "fact": {"type": "string", "description": "The specific fact to remember."},
        },
        "required": ["fact"],
    }

    @classmethod
    def default_spec(cls, model: str) -> GeminiToolSpec:
        del model
        return GEMINI_MEMORY_SPEC

    def to_params(self) -> genai_types.Tool:
        return _decl(self.name, self.description, self.parameters)

    async def execute(self, arguments: dict[str, Any]) -> MCPToolResult:
        fact = arguments.get("fact")
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("fact is required")
        text = fact.strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return await self.file_write(f"/memories/gemini-{digest}.md", f"{text}\n")


__all__ = ["GeminiMemoryTool"]
