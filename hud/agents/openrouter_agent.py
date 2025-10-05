"""OpenRouter MCP Agent implementation.

This agent provides intelligent support for both OpenAI and Anthropic models
available through OpenRouter's unified API, adapting tool formats and request
handling based on the selected model.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from hud.datasets import Task

from hud import instrument
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

from .openai_chat_generic import GenericOpenAIChatAgent

logger = logging.getLogger(__name__)


class OpenRouterAgent(GenericOpenAIChatAgent):
    """MCP-enabled agent that uses OpenRouter's unified API."""

    metadata: ClassVar[dict[str, Any]] = {}

    # Default OpenRouter base URL
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    # Model prefixes for type detection
    ANTHROPIC_PREFIXES = ("anthropic/", "claude-")
    OPENAI_PREFIXES = ("openai/", "gpt-")

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
        site_url: str | None = None,
        app_name: str | None = None,
        completion_kwargs: dict[str, Any] | None = None,
        max_retries: int = 3,
        enable_prompt_caching: bool = True,
        **agent_kwargs: Any,
    ) -> None:
        """Initialize OpenRouter agent."""
        headers = {
            "HTTP-Referer": site_url or "https://hud-ai.dev",
            "X-Title": app_name or "HUD Agent",
        }

        openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or self.DEFAULT_BASE_URL,
            default_headers=headers,
            max_retries=max_retries,
        )

        super().__init__(
            openai_client=openai_client,
            model_name=model_name,
            completion_kwargs=completion_kwargs or {},
            **agent_kwargs,
        )

        self.model_type = self._detect_model_type(model_name)
        self.max_retries = max_retries
        self.enable_prompt_caching = enable_prompt_caching
        self._cache_control = {"type": "ephemeral"}
        self._prompt_cache_applied = False

        logger.info(
            "OpenRouterAgent initialized for %s model: %s",
            self.model_type,
            model_name,
        )

    async def initialize(self, task: str | "Task" | None = None) -> None:
        """Reset prompt cache state before each run."""
        self._prompt_cache_applied = False
        await super().initialize(task)

    def _detect_model_type(self, model_name: str) -> str:
        """Detect whether model is OpenAI or Anthropic based on name."""
        model_lower = model_name.lower()

        if any(model_lower.startswith(prefix) for prefix in self.ANTHROPIC_PREFIXES):
            return "anthropic"
        if any(model_lower.startswith(prefix) for prefix in self.OPENAI_PREFIXES):
            return "openai"

        logger.warning(
            "Could not detect model type for '%s', defaulting to OpenAI format",
            model_name,
        )
        return "unknown"

    # -----------------------------
    # Tool filtering helpers
    # -----------------------------
    def _is_function_tool(self, schema: dict) -> bool:
        return schema.get("type") == "function"

    def _function_name(self, schema: dict) -> str:
        return (schema.get("function") or {}).get("name", "")

    def _is_playwright_fn(self, schema: dict) -> bool:
        return self._is_function_tool(schema) and self._function_name(schema) == "playwright"

    def _is_anthropic_computer(self, schema: dict) -> bool:
        # Anthropic Computer Use tool is a non-function tool with explicit type
        return schema.get("type") == "computer_20250124"

    def _is_qwen_computer(self, schema: dict) -> bool:
        # Qwen computer tool type (should not be passed to OpenAI/Anthropic)
        return schema.get("type") == "computer_use"

    def _is_openai_or_generic_computer_fn(self, schema: dict) -> bool:
        if not self._is_function_tool(schema):
            return False
        name = self._function_name(schema)
        return name in {"openai_computer", "computer"}

    def _filter_tools_for_model(self, base_schemas: list[dict]) -> list[dict]:
        """
        Filter the available tools to match the provider behind the chosen model.
        This prevents exposing incompatible computer tools to a given provider.
        """
        if self.model_type == "anthropic":
            filtered: list[dict] = []
            added_anthropic_computer = False

            # Prefer the Anthropic computer tool; allow other *function* tools,
            # but block OpenAI/generic computer function tools and Qwen computer
            # tool. Keep playwright/setup/evaluate/etc (they are function tools).
            for schema in base_schemas:
                if self._is_anthropic_computer(schema):
                    if not added_anthropic_computer:
                        filtered.append(schema)
                        added_anthropic_computer = True
                    continue

                if self._is_qwen_computer(schema):
                    continue

                if self._is_openai_or_generic_computer_fn(schema):
                    continue

                filtered.append(schema)

            logger.info(
                "Filtered tools for Anthropic: kept %d of %d",
                len(filtered),
                len(base_schemas),
            )
            return filtered

        if self.model_type == "openai":
            function_tools = [s for s in base_schemas if self._is_function_tool(s)]
            have_openai_computer = any(
                self._function_name(s) == "openai_computer" for s in function_tools
            )

            # OpenAI Chat Completions only accepts function tools. Drop
            # Anthropic/Qwen computer tools and any other non-function tool types.
            filtered: list[dict] = []
            for s in function_tools:
                # Avoid the Playwright function tool for OpenAI models because the
                # subclass overrides (e.g., PlaywrightToolWithMemory.click) define
                # defaults using pydantic.Field(...). When invoked without explicit
                # args, those FieldInfo defaults leak into Playwright and cause:
                # "Object of type FieldInfo is not JSON serializable".
                # We route OpenAI models to the Computer Use tool instead.
                if self._is_playwright_fn(s):
                    continue
                fname = self._function_name(s)
                # If we have a provider-specific openai_computer tool, drop the generic "computer"
                if have_openai_computer and fname == "computer":
                    continue
                filtered.append(s)

            logger.info(
                "Filtered tools for OpenAI: kept %d of %d (function-only)",
                len(filtered),
                len(base_schemas),
            )
            return filtered

        # Unknown model type — do not filter (preserves current behavior).
        logger.warning("Unknown model type; not filtering tool schemas")
        return base_schemas

    def get_tool_schemas(self) -> list[dict]:
        """Get tool schemas adapted for the model type."""
        base_schemas = super().get_tool_schemas()
        # Filter the tool list based on the target provider so we only expose
        # compatible tools.
        filtered_schemas = self._filter_tools_for_model(base_schemas)

        if self.model_type == "anthropic":
            for schema in filtered_schemas:
                if "function" in schema:
                    func = schema["function"]
                    if "parameters" not in func:
                        func["parameters"] = {"type": "object", "properties": {}}

                    params = func["parameters"]
                    if "required" not in params and "properties" in params:
                        params["required"] = list(params["properties"].keys())

        return filtered_schemas

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: list[Any]) -> AgentResponse:
        """Get response from OpenRouter with intelligent retry logic."""
        tool_schemas = self.get_tool_schemas()

        protected_keys = {"model", "messages", "tools", "parallel_tool_calls"}
        extra = {
            k: v
            for k, v in (self.completion_kwargs or {}).items()
            if k not in protected_keys
        }

        if self.model_type == "anthropic":
            extra.setdefault("parallel_tool_calls", False)

        attempt = 0
        last_error: Exception | None = None

        while attempt < self.max_retries:
            attempt += 1

            try:
                response = await self.oai.chat.completions.create(  # type: ignore[union-attr]
                    model=self.model_name,
                    messages=messages,
                    tools=tool_schemas or None,
                    **extra,
                )

                choice = response.choices[0]
                msg = choice.message

                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    assistant_msg["content"] = msg.content
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = msg.tool_calls

                messages.append(assistant_msg)

                tool_calls: list[MCPToolCall] = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.function.name is not None:
                            tool_calls.append(self._oai_to_mcp(tc))

                done = choice.finish_reason in {"stop", "length", "end_turn"}

                if choice.finish_reason and choice.finish_reason not in {
                    "stop",
                    "length",
                    "end_turn",
                    "tool_calls",
                }:
                    logger.warning(
                        "Unusual finish_reason from OpenRouter: %s",
                        choice.finish_reason,
                    )

                return AgentResponse(
                    content=msg.content or "",
                    tool_calls=tool_calls,
                    done=done,
                    raw=response,
                )

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                error_str = str(exc).lower()

                if "404" in error_str or "no endpoints found" in error_str:
                    logger.error(
                        "OpenRouter provider error (attempt %d/%d): No provider supports tool calling for %s",
                        attempt,
                        self.max_retries,
                        self.model_name,
                    )

                    if attempt >= self.max_retries:
                        return AgentResponse(
                            content=(
                                "Model "
                                f"{self.model_name} does not support tool calling via OpenRouter. "
                                "Try anthropic/claude-3.5-sonnet:beta or openai/gpt-4o"
                            ),
                            tool_calls=[],
                            done=True,
                            isError=True,
                            raw=None,
                        )

                elif "invalid json" in error_str or "malformed" in error_str:
                    logger.warning(
                        "JSON parsing error (attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        exc,
                    )

                    if attempt >= self.max_retries:
                        return AgentResponse(
                            content="Model generated invalid JSON. Response was truncated.",
                            tool_calls=[],
                            done=True,
                            isError=True,
                            raw=None,
                        )

                else:
                    logger.error(
                        "OpenRouter request failed (attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        exc,
                    )

                if attempt < self.max_retries:
                    await asyncio.sleep(1.0)

        err_msg = (
            f"Request failed after {self.max_retries} attempts: {last_error}"  # type: ignore[str-format]
        )
        return AgentResponse(
            content=err_msg,
            tool_calls=[],
            done=True,
            isError=True,
            raw=None,
        )

    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[Any]:
        """Format tool results for OpenRouter."""
        formatted = await super().format_tool_results(tool_calls, tool_results)

        if self.model_type == "anthropic":
            for msg in formatted:
                if msg.get("role") == "tool":
                    if not isinstance(msg.get("content"), str):
                        msg["content"] = str(msg.get("content", ""))
                    if "tool_call_id" not in msg:
                        logger.warning(
                            "Tool message missing tool_call_id for Anthropic model",
                        )

        return formatted

    async def get_system_messages(self) -> list[Any]:
        """Include Anthropic cache_control breakpoints when enabled."""
        messages = await super().get_system_messages()

        if not (
            self.enable_prompt_caching
            and self.model_type == "anthropic"
            and messages
        ):
            return messages

        cached_messages: list[Any] = []
        for msg in messages:
            if not isinstance(msg, dict):
                cached_messages.append(msg)
                continue

            content = msg.get("content")
            new_msg = dict(msg)

            if isinstance(content, str):
                new_msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": dict(self._cache_control),
                    }
                ]
            elif isinstance(content, list):
                cached_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        updated_part = dict(part)
                        updated_part.setdefault(
                            "cache_control", dict(self._cache_control)
                        )
                        cached_content.append(updated_part)
                    else:
                        cached_content.append(part)
                new_msg["content"] = cached_content

            cached_messages.append(new_msg)

        return cached_messages

    async def format_blocks(self, blocks: list[Any]) -> list[Any]:
        """Add cache_control to the first user message when applicable."""
        messages = await super().format_blocks(blocks)

        if not (
            self.enable_prompt_caching
            and self.model_type == "anthropic"
            and not self._prompt_cache_applied
        ):
            return messages

        cached_messages: list[Any] = []
        cache_applied = False

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    updated_content = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            updated_part = dict(part)
                            updated_part.setdefault(
                                "cache_control", dict(self._cache_control)
                            )
                            updated_content.append(updated_part)
                            cache_applied = True
                        else:
                            updated_content.append(part)
                    new_msg = dict(msg)
                    new_msg["content"] = updated_content
                    cached_messages.append(new_msg)
                    continue
            cached_messages.append(msg)

        if cache_applied:
            self._prompt_cache_applied = True
            return cached_messages

        return messages

    @staticmethod
    def _oai_to_mcp(tool_call: Any) -> MCPToolCall:  # type: ignore[valid-type]
        """Convert OpenAI tool_call to MCPToolCall."""
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse tool arguments as JSON: %s",
                tool_call.function.arguments,
            )
            args = {}

        if isinstance(args, list):
            args = args[0] if args else {}

        if not isinstance(args, dict):
            logger.warning("Tool arguments not a dict: %s", type(args))
            args = {}

        return MCPToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=args,
        )
