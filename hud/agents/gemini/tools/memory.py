"""Agent-side Gemini memory tool."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from hud.agents.tools.base import CallTool
    from hud.types import MCPToolResult

from .base import GeminiTool, GeminiToolSpec

GEMINI_MEMORY_SPEC = GeminiToolSpec(api_type="save_memory", api_name="save_memory")


class GeminiMemoryTool(GeminiTool):
    """Translate Gemini save_memory calls into the file-backed memory env primitive."""

    name = "save_memory"
    capability = "memory"
    description = "Saves a specific fact to long-term memory."
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

    async def execute(self, call_tool: CallTool, arguments: dict[str, Any]) -> MCPToolResult:
        fact = arguments.get("fact")
        if not isinstance(fact, str) or not fact.strip():
            raise ValueError("fact is required")
        text = fact.strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return await super().execute(
            call_tool,
            {
                "command": "create",
                "path": f"/memories/gemini-{digest}.md",
                "file_text": f"{text}\n",
            },
        )
