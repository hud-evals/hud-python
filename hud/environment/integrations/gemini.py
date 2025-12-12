"""Google/Gemini integrations - format conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mcp.types as mcp_types

__all__ = ["GeminiMixin"]


class GeminiMixin:
    """Mixin providing Google/Gemini format conversion.

    Format methods (no deps):
        as_gemini_tools() - Gemini tool format
        as_gemini_tool_config() - Tool execution config

    Requires: as_tools() -> list[mcp_types.Tool]
    """

    def as_tools(self) -> list[mcp_types.Tool]:
        raise NotImplementedError

    def as_gemini_tools(self) -> list[dict[str, Any]]:
        """Convert to Gemini/Google AI tool format.

        Returns:
            List with function_declarations for Gemini API.

        Example:
            ```python
            import google.generativeai as genai

            model = genai.GenerativeModel("gemini-1.5-pro")
            async with env:
                response = model.generate_content(
                    "Navigate to google.com",
                    tools=env.as_gemini_tools(),
                )
                # Execute tool calls
                for part in response.candidates[0].content.parts:
                    if fn := part.function_call:
                        result = await env.call_tool(part)
            ```
        """
        return [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {"type": "object", "properties": {}},
                    }
                    for t in self.as_tools()
                ]
            }
        ]

    def as_gemini_tool_config(
        self,
        mode: str = "AUTO",
        allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get Gemini tool_config for controlling tool execution.

        Args:
            mode: "AUTO", "ANY", or "NONE"
            allowed_tools: If mode is "ANY", list of allowed tool names

        Returns:
            Tool config dict for Gemini API.

        Example:
            ```python
            import google.generativeai as genai

            model = genai.GenerativeModel("gemini-1.5-pro")
            async with env:
                # Force specific tool usage
                response = model.generate_content(
                    "Search for cats",
                    tools=env.as_gemini_tools(),
                    tool_config=env.as_gemini_tool_config(mode="ANY", allowed_tools=["search"]),
                )
            ```
        """
        config: dict[str, Any] = {"function_calling_config": {"mode": mode}}
        if mode == "ANY" and allowed_tools:
            config["function_calling_config"]["allowed_function_names"] = allowed_tools
        return config
