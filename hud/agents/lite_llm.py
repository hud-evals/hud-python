"""LiteLLM MCP Agent implementation.

Same OpenAI chat-completions shape + MCP tool plumbing,
but transport is LiteLLM and (optionally) tools are shaped by LiteLLM's MCP transformer.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import litellm

from .openai_chat_generic import GenericOpenAIChatAgent

logger = logging.getLogger(__name__)

# Prefer LiteLLM's built-in MCP -> OpenAI tool transformer (handles Bedrock nuances)
try:
    from litellm.experimental_mcp_client.tools import (
        transform_mcp_tool_to_openai_tool,
    )
except Exception:  # pragma: no cover - optional dependency
    transform_mcp_tool_to_openai_tool = None  # type: ignore


class LiteAgent(GenericOpenAIChatAgent):
    """
    Same OpenAI chat-completions shape + MCP tool plumbing,
    but transport is LiteLLM and (optionally) tools are shaped by LiteLLM's MCP transformer.
    """

    metadata: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        completion_kwargs: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        # We don't need an OpenAI client; pass None
        super().__init__(
            openai_client=None,
            model_name=model_name,
            completion_kwargs=completion_kwargs,
            **agent_kwargs,
        )

    def get_tool_schemas(self) -> list[dict]:
        # Prefer LiteLLM's stricter transformer (handles Bedrock & friends)
        if transform_mcp_tool_to_openai_tool is not None:
            return [
                transform_mcp_tool_to_openai_tool(t)  # returns ChatCompletionToolParam-like dict
                for t in self.get_available_tools()
            ]
        # Fallback to the generic OpenAI sanitizer
        return GenericOpenAIChatAgent.get_tool_schemas(self)

    async def _invoke_chat_completion(
        self,
        *,
        messages: list[Any],
        tools: list[dict] | None,
        extra: dict[str, Any],
    ) -> Any:
        return await litellm.acompletion(
            model=self.model_name,
            messages=messages,
            tools=tools or None,  # LiteLLM tolerates None better than []
            **extra,
        )
