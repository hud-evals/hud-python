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
            try:
                tools = [
                    transform_mcp_tool_to_openai_tool(t)  # returns ChatCompletionToolParam-like dict
                    for t in self.get_available_tools()
                ]
                sanitized_tools = [self._sanitize_for_openrouter(tool) for tool in tools]
                self.hud_console.debug(
                    f"Using LiteLLM MCP transformer with OpenRouter fixes for {len(sanitized_tools)} tools"
                )
                return sanitized_tools
            except Exception as transform_err:  # pragma: no cover - best effort logging
                self.hud_console.warning(
                    f"LiteLLM transformer failed: {transform_err}, falling back to generic sanitizer"
                )
        # Fallback to the generic OpenAI sanitizer
        tools = GenericOpenAIChatAgent.get_tool_schemas(self)
        self.hud_console.debug(f"Using HUD generic OpenAI transformer for {len(tools)} tools")
        return tools

    def _sanitize_for_openrouter(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Sanitize tool schemas to satisfy stricter OpenRouter/Azure validation."""
        tool_copy = copy.deepcopy(tool)

        if isinstance(tool_copy, dict):
            function_block = tool_copy.get("function")
            if isinstance(function_block, dict) and "parameters" in function_block:
                function_block["parameters"] = self._fix_array_schemas(function_block["parameters"])

        return tool_copy

    def _fix_array_schemas(self, schema: Any) -> Any:
        """Recursively ensure array schemas include an items definition."""
        if isinstance(schema, list):
            return [self._fix_array_schemas(item) for item in schema]

        if not isinstance(schema, dict):
            return schema

        fixed: dict[str, Any] = {}
        for key, value in schema.items():
            if key in {"anyOf", "allOf", "oneOf"} and isinstance(value, list):
                fixed[key] = [self._fix_array_schemas(variant) for variant in value]
            elif key in {"properties", "patternProperties"} and isinstance(value, dict):
                fixed[key] = {
                    property_key: self._fix_array_schemas(property_value)
                    for property_key, property_value in value.items()
                }
            elif key in {"items", "prefixItems"} and isinstance(value, list):
                fixed[key] = [self._fix_array_schemas(item) for item in value]
            elif key == "items" and isinstance(value, dict):
                fixed[key] = self._fix_array_schemas(value)
            elif isinstance(value, dict):
                fixed[key] = self._fix_array_schemas(value)
            else:
                fixed[key] = value

        if fixed.get("type") == "array" and "items" not in fixed:
            prefix_items = fixed.get("prefixItems")
            if isinstance(prefix_items, list) and prefix_items:
                first_prefix = prefix_items[0]
                if isinstance(first_prefix, dict):
                    inferred_items = copy.deepcopy(first_prefix)
                    inferred_items.setdefault("type", "string")
                else:
                    inferred_items = {"type": "string"}
            else:
                inferred_items = {"type": "string"}

            fixed["items"] = inferred_items

        return fixed

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
            error_msg = f"LiteLLM completion failed: {e}"

            task_run_id = None
            try:
                from hud.otel.context import get_current_task_run_id

                task_run_id = get_current_task_run_id()
            except Exception as context_err:
                logger.debug(
                    "Failed to fetch task context for LiteLLM error logging: %s",
                    context_err,
                )

            if task_run_id:
                record = logging.LogRecord(
                    name=logger.name,
                    level=logging.ERROR,
                    pathname=__file__,
                    lineno=0,
                    msg=error_msg,
                    args=(),
                    exc_info=(type(e), e, e.__traceback__),
                )
                record.hud_task_run_id = task_run_id
                logger.handle(record)
            else:
                logger.error(error_msg, exc_info=(type(e), e, e.__traceback__))

            self.hud_console.error(error_msg)
            raise
