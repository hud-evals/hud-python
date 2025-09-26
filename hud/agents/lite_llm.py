"""LiteLLM MCP Agent implementation.
Same OpenAI chat-completions shape + MCP tool plumbing,
but transport is LiteLLM and (optionally) tools are shaped by LiteLLM's MCP transformer.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, ClassVar

import litellm

from .openai_chat_generic import GenericOpenAIChatAgent
from ..utils.hud_console import HUDConsole

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

        # Initialize HUD console for better logging
        self.hud_console = HUDConsole(logger=logger)
        self.hud_console.debug(f"LiteLLM agent initialized with model: {model_name}")

        if completion_kwargs:
            self.hud_console.debug(f"Completion kwargs: {completion_kwargs}")

    def get_tool_schemas(self) -> list[dict]:
        # Prefer LiteLLM's stricter transformer (handles Bedrock & friends)
        if transform_mcp_tool_to_openai_tool is not None:
            tools = [
                transform_mcp_tool_to_openai_tool(t)  # returns ChatCompletionToolParam-like dict
                for t in self.get_available_tools()
            ]
            self.hud_console.debug(f"Using LiteLLM MCP transformer for {len(tools)} tools")
            return tools
        # Fallback to the generic OpenAI sanitizer
        tools = GenericOpenAIChatAgent.get_tool_schemas(self)
        self.hud_console.debug(f"Using generic OpenAI transformer for {len(tools)} tools")
        return tools

    def _add_prompt_caching(self, messages: list[dict]) -> list[dict]:
        """Add prompt caching to messages for models that support it."""
        # Check if model supports prompt caching
        try:
            from litellm.utils import supports_prompt_caching
            if not supports_prompt_caching(self.model_name):
                self.hud_console.debug(f"Model {self.model_name} does not support prompt caching, skipping")
                return messages
        except (ImportError, AttributeError):
            # If function not available, apply caching universally and let LiteLLM handle it
            self.hud_console.debug("litellm.utils.supports_prompt_caching not available, applying caching universally")

        self.hud_console.debug(f"Adding prompt caching to messages for model: {self.model_name}")
        messages_cached = copy.deepcopy(messages)

        # Mark last user message with cache control
        if (
            messages_cached
            and isinstance(messages_cached[-1], dict)
            and messages_cached[-1].get("role") == "user"
        ):
            content = messages_cached[-1].get("content")

            # Handle string content - convert to list format for caching
            if isinstance(content, str):
                messages_cached[-1]["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
                self.hud_console.debug("Added caching to string content (converted to list format)")
            # Handle list content (already in structured format)
            elif isinstance(content, list):
                cached_blocks = 0
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type in ["text", "image", "tool_use", "tool_result"]:
                            block["cache_control"] = {"type": "ephemeral"}
                            cached_blocks += 1
                self.hud_console.debug(f"Added caching to {cached_blocks} content blocks")

        return messages_cached

    def _add_tools_caching(self, tools: list[dict] | None) -> list[dict] | None:
        """Add caching to tools for models that support it."""
        if not tools:
            return tools

        # Check if model supports prompt caching (which also covers tools caching)
        try:
            from litellm.utils import supports_prompt_caching
            if not supports_prompt_caching(self.model_name):
                self.hud_console.debug(f"Model {self.model_name} does not support tools caching, skipping")
                return tools
        except (ImportError, AttributeError):
            # If function not available, apply caching universally and let LiteLLM handle it
            self.hud_console.debug("litellm.utils.supports_prompt_caching not available, applying tools caching universally")

        self.hud_console.debug(f"Adding caching to {len(tools)} tools for model: {self.model_name}")
        tools_cached = copy.deepcopy(tools)

        # Add cache control to function definitions
        cached_tools = 0
        for tool in tools_cached:
            if isinstance(tool, dict) and tool.get("type") == "function":
                function = tool.get("function", {})
                if isinstance(function, dict):
                    function["cache_control"] = {"type": "ephemeral"}
                    cached_tools += 1

        self.hud_console.debug(f"Added caching to {cached_tools} function tools")
        return tools_cached

    async def _invoke_chat_completion(
        self,
        *,
        messages: list[Any],
        tools: list[dict] | None,
        extra: dict[str, Any],
    ):
        self.hud_console.debug(f"Invoking LiteLLM completion with model: {self.model_name}")
        self.hud_console.debug(f"Messages count: {len(messages)}")
        self.hud_console.debug(f"Tools count: {len(tools) if tools else 0}")

        if extra:
            self.hud_console.debug(f"Extra parameters: {list(extra.keys())}")

        # Apply caching to messages and tools
        messages_cached = self._add_prompt_caching(messages)
        tools_cached = self._add_tools_caching(tools)

        try:
            self.hud_console.debug("Calling litellm.acompletion...")
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages_cached,
                tools=tools_cached or None,  # LiteLLM tolerates None better than []
                **extra,
            )

            # Log response details
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.hud_console.debug(f"Token usage - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}")

                # Log cache statistics if available
                if hasattr(usage, 'cache_read_input_tokens') and usage.cache_read_input_tokens:
                    self.hud_console.info(f"🔄 Cache read tokens: {usage.cache_read_input_tokens}")
                if hasattr(usage, 'cache_creation_input_tokens') and usage.cache_creation_input_tokens:
                    self.hud_console.info(f"💾 Cache creation tokens: {usage.cache_creation_input_tokens}")

            self.hud_console.debug("LiteLLM completion successful")
            return response

        except Exception as e:
            self.hud_console.error(f"LiteLLM completion failed: {e}")
            raise
