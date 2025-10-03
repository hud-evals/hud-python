"""LiteLLM-powered MCP Agent.

Provides the same MCP behaviour as hud/agents/claude.py / openai_chat_generic.py
while routing through LiteLLM. Supports Anthropic computer-use (when available),
prompt caching, and OpenAI-style tool result formatting so LiteLLM can talk to
providers such as OpenRouter or Anthropic directly.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Sequence, cast

import litellm
from litellm.exceptions import APIError, BadRequestError
import mcp.types as types

import hud
from hud.settings import settings
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole

from .base import MCPAgent

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaContentBlockParam,
        BetaImageBlockParam,
        BetaMessageParam,
        BetaTextBlockParam,
        BetaToolResultBlockParam,
    )
else:  # pragma: no cover - fallback when anthropic types are absent at runtime
    BetaCacheControlEphemeralParam = dict  # type: ignore
    BetaContentBlockParam = dict  # type: ignore
    BetaImageBlockParam = dict  # type: ignore
    BetaMessageParam = dict  # type: ignore
    BetaTextBlockParam = dict  # type: ignore
    BetaToolResultBlockParam = dict  # type: ignore

try:  # pragma: no cover - optional import
    from pydantic.fields import FieldInfo as _PydanticFieldInfo
    from pydantic_core import PydanticUndefined
except Exception:  # pragma: no cover - fallback when pydantic not present
    _PydanticFieldInfo = None  # type: ignore
    PydanticUndefined = object()  # type: ignore

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "anthropic/claude-sonnet-4": "anthropic/claude-sonnet-4-20250514",
}


class LiteLLMAgent(MCPAgent):
    """Generic MCP agent implemented on top of LiteLLM."""

    metadata: ClassVar[dict[str, Any]] = {
        "display_width": computer_settings.ANTHROPIC_COMPUTER_WIDTH,
        "display_height": computer_settings.ANTHROPIC_COMPUTER_HEIGHT,
    }

    def __init__(
        self,
        *,
        model: str = "anthropic/claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        use_computer_beta: bool = True,
        completion_kwargs: dict[str, Any] | None = None,
        cache_control_injection: Sequence[str] | None = None,
        validate_api_key: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model = model
        self.model_name = self.model
        self.max_tokens = max_tokens
        self.use_computer_beta = use_computer_beta
        self.completion_kwargs = completion_kwargs or {}
        self.hud_console = HUDConsole(logger=logger)
        self._resolved_completion_kwargs: dict[str, Any] = dict(self.completion_kwargs)
        self._transport_provider: str | None = None
        self._litellm_model = self.model
        self._validate_api_key = validate_api_key
        self._cache_control_injection = list(cache_control_injection or ["system", "last_user"])
        self._anthropic_key_validated = False
        self._computer_tool_type: str | None = None

        self._claude_to_mcp_tool_map: dict[str, str] = {}
        self.claude_tools: list[dict] = []

        claude_instructions = """
        You are Claude, an AI assistant created by Anthropic. You are helpful, harmless, and honest.

        When working on tasks:
        1. Be thorough and systematic in your approach
        2. Complete tasks autonomously without asking for confirmation
        3. Use available tools efficiently to accomplish your goals
        4. Verify your actions and ensure task completion
        5. Be precise and accurate in all operations

        Remember: You are expected to complete tasks autonomously. The user trusts you to accomplish what they asked.
        """.strip()

        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{claude_instructions}"
        else:
            self.system_prompt = claude_instructions

        self._resolve_transport_layer()

    async def initialize(self, task: Any = None) -> None:
        await super().initialize(task)
        self._convert_tools_for_claude()

    async def get_system_messages(self) -> list[BetaMessageParam]:
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[BetaMessageParam]:
        anthropic_blocks: list[BetaContentBlockParam] = []

        for block in blocks:
            if isinstance(block, types.TextContent):
                anthropic_blocks.append(
                    cast(
                        "BetaTextBlockParam",
                        {
                            "type": "text",
                            "text": block.text,
                        },
                    )
                )
            elif isinstance(block, types.ImageContent):
                anthropic_blocks.append(
                    cast(
                        "BetaImageBlockParam",
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.mimeType,
                                "data": block.data,
                            },
                        },
                    )
                )
            else:
                anthropic_blocks.append(cast("BetaContentBlockParam", block))

        return [
            cast(
                "BetaMessageParam",
                {
                    "role": "user",
                    "content": anthropic_blocks,
                },
            )
        ]

    @hud.instrument(span_type="agent", record_args=False, record_result=True)
    async def get_response(self, messages: list[BetaMessageParam]) -> AgentResponse:
        current_messages = list(messages)

        while True:
            messages_cached = self._add_prompt_caching(current_messages)
            request_kwargs = self._build_litellm_request(messages_cached)

            try:
                response = await litellm.acompletion(**request_kwargs)
                break
            except BadRequestError as e:
                if self._maybe_adjust_computer_tool(e):
                    continue

                error_text = f"LiteLLM BadRequestError: {e}"
                self.hud_console.error(error_text)
                return AgentResponse(
                    content=error_text,
                    tool_calls=[],
                    done=True,
                    isError=True,
                    raw=e,
                )
            except APIError as e:
                error_text = f"LiteLLM APIError: {e}"
                self.hud_console.error(error_text)
                return AgentResponse(
                    content=error_text,
                    tool_calls=[],
                    done=True,
                    isError=True,
                    raw=e,
                )
            except Exception as e:
                msg = str(e).lower()
                too_long = (
                    "prompt is too long" in msg
                    or "request_too_large" in msg
                    or "413" in msg
                )
                if too_long and len(current_messages) > 21:
                    self.hud_console.warning(
                        "Prompt too long via LiteLLM; truncating message history"
                    )
                    current_messages = [current_messages[0], *current_messages[-20:]]
                    continue
                raise

        message_obj, finish_reason = _extract_message_and_finish_reason(response)

        assistant_payload: dict[str, Any] = {
            "role": _get_field(message_obj, "role", "assistant"),
        }

        content_blocks = _get_field(message_obj, "content")
        messages.append(assistant_payload)

        tool_calls_seen: set[str | None] = set()
        result = AgentResponse(content="", tool_calls=[], done=True, raw=response)
        assistant_tool_calls_payload: list[dict[str, Any]] = []

        openai_style_tool_calls = _get_field(message_obj, "tool_calls")
        if openai_style_tool_calls:
            for tc in openai_style_tool_calls:
                tc_id = _get_field(tc, "id")
                func_spec = _get_field(tc, "function", {})
                try:
                    arguments = _safe_json_loads(_get_field(func_spec, "arguments"))
                except Exception:
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                arguments = _to_json_safe(arguments)
                name = _get_field(func_spec, "name")
                if isinstance(name, str) and name in self._claude_to_mcp_tool_map:
                    name = self._claude_to_mcp_tool_map[name]
                result.tool_calls.append(
                    MCPToolCall(id=tc_id, name=name, arguments=arguments)
                )
                tool_calls_seen.add(tc_id)
                result.done = False
                assistant_tool_calls_payload.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": _get_field(func_spec, "name"),
                            "arguments": json.dumps(arguments),
                        },
                    }
                )

        text_content = ""
        thinking_content = ""

        if isinstance(content_blocks, list):
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_name = block.get("name", "")
                    mcp_name = self._claude_to_mcp_tool_map.get(tool_name, tool_name)
                    tool_id = block.get("id")
                    if tool_id in tool_calls_seen:
                        continue
                    tool_input = _to_json_safe(block.get("input", {}))
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    result.tool_calls.append(
                        MCPToolCall(
                            id=tool_id,
                            name=mcp_name,
                            arguments=tool_input,
                            claude_name=tool_name,
                        )
                    )
                    tool_calls_seen.add(tool_id)
                    result.done = False
                    assistant_tool_calls_payload.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input),
                            },
                        }
                    )
                elif block_type == "text":
                    text_content += block.get("text", "")
                elif block_type == "thinking":
                    thinking_content += f"Thinking: {block.get('thinking', '')}\n"
        elif isinstance(content_blocks, str):
            text_content += content_blocks

        if thinking_content:
            result.content = thinking_content + text_content
        else:
            result.content = text_content

        if finish_reason == "tool_use":
            result.done = False

        assistant_text = (thinking_content + text_content).strip()
        assistant_payload["content"] = assistant_text if assistant_text else ""
        if assistant_tool_calls_payload:
            assistant_payload["tool_calls"] = assistant_tool_calls_payload

        return result

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        image_parts: list[dict[str, Any]] = []

        for call, res in zip(tool_calls, tool_results, strict=False):
            tool_call_id = call.id
            if not tool_call_id:
                self.hud_console.warning(f"No tool_call_id for result from {call.name}")
                continue

            text_fragments: list[str] = []
            content_items: Sequence[Any]
            if res.content:
                content_items = res.content
            elif res.structuredContent:
                content_items = [res.structuredContent.get("result", "")]
            else:
                content_items = []

            if res.isError:
                error_msg = "Tool execution failed"
                for item in content_items:
                    if isinstance(item, types.TextContent):
                        error_msg = item.text
                        break
                    if isinstance(item, dict) and item.get("type") == "text":
                        error_msg = item.get("text", error_msg)
                        break
                text_fragments.append(f"Error: {error_msg}")
            else:
                for item in content_items:
                    if isinstance(item, types.TextContent):
                        text_fragments.append(item.text)
                    elif isinstance(item, types.ImageContent):
                        mime = getattr(item, "mimeType", "image/png")
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{item.data}",
                                },
                            }
                        )
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            text_fragments.append(item.get("text", ""))
                        elif item.get("type") == "image":
                            mime = item.get("mimeType", "image/png")
                            data = item.get("data", "")
                            image_parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime};base64,{data}",
                                    },
                                }
                            )

            content_text = "\n".join(fragment for fragment in text_fragments if fragment).strip()
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content_text or "",
                }
            )

        if image_parts:
            rendered.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Tool returned the following images:"},
                        *image_parts,
                    ],
                }
            )

        return rendered

    async def create_user_message(self, text: str) -> BetaMessageParam:
        return cast("BetaMessageParam", {"role": "user", "content": text})

    def _convert_tools_for_claude(self) -> list[dict]:
        claude_tools: list[dict] = []
        self._claude_to_mcp_tool_map = {}

        computer_tool_priority = ["anthropic_computer", "computer_anthropic", "computer"]
        selected_computer_tool = None

        for priority_name in computer_tool_priority:
            for tool in self._available_tools:
                if tool.name == priority_name or tool.name.endswith(f"_{priority_name}"):
                    selected_computer_tool = tool
                    break
            if selected_computer_tool:
                break

        self._computer_tool_type = None

        if selected_computer_tool:
            claude_tool = {
                "type": self._choose_computer_tool_type(),
                "name": "computer",
                "display_width_px": self.metadata["display_width"],
                "display_height_px": self.metadata["display_height"],
            }
            self._computer_tool_type = claude_tool["type"]
            self._claude_to_mcp_tool_map["computer"] = selected_computer_tool.name
            claude_tools.append(claude_tool)
            self.hud_console.debug(
                f"Using {selected_computer_tool.name} as computer tool for Claude"
            )

        for tool in self._available_tools:
            is_computer_tool = any(
                tool.name == priority_name or tool.name.endswith(f"_{priority_name}")
                for priority_name in computer_tool_priority
            )
            if is_computer_tool or tool.name in self.lifecycle_tools:
                continue

            claude_tool = {
                "name": tool.name,
                "description": tool.description or f"Execute {tool.name}",
                "input_schema": tool.inputSchema
                or {
                    "type": "object",
                    "properties": {},
                },
            }
            self._claude_to_mcp_tool_map[tool.name] = tool.name
            claude_tools.append(claude_tool)

        self.claude_tools = claude_tools
        return claude_tools

    def _choose_computer_tool_type(self) -> str:
        if self._transport_provider == "openrouter":
            return "computer_20250124"
        return "computer_20241022"

    def _is_openrouter_model(self) -> bool:
        return self._transport_provider == "openrouter"

    def _build_litellm_request(self, messages: list[BetaMessageParam]) -> dict[str, Any]:
        protected = {"model", "messages", "tools", "tool_choice", "max_tokens"}
        extra = {k: v for k, v in self._resolved_completion_kwargs.items() if k not in protected}

        tools_payload = self._prepare_tools_for_transport()

        model_name = self._litellm_model
        if self._transport_provider == "anthropic" and not model_name.startswith("anthropic/"):
            model_name = f"anthropic/{model_name}"

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "tools": tools_payload,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "drop_params": True,
            **extra,
        }

        if self._is_openrouter_model():
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False
        else:
            kwargs["tool_choice"] = "auto"

        if not self._is_openrouter_model():
            beta_headers: set[str] = {"prompt-caching-2024-07-31"}
            if self.use_computer_beta:
                if self._computer_tool_type == "computer_20250124":
                    beta_headers.add("computer-use-2025-01-24")
                else:
                    beta_headers.add("computer-use-2024-10-22")

            extra_headers: dict[str, str] = kwargs.setdefault("extra_headers", {})  # type: ignore[assignment]
            existing = extra_headers.get("anthropic-beta")
            if existing:
                beta_headers.update(filter(None, existing.split(",")))
            extra_headers["anthropic-beta"] = ",".join(sorted(beta_headers))

        return kwargs

    def _prepare_tools_for_transport(self) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for tool in self.claude_tools:
            tool_type = tool.get("type")
            name = tool.get("name")
            description = tool.get("description")
            input_schema = tool.get("input_schema")

            if isinstance(tool_type, str) and tool_type.startswith("computer_"):
                params = {
                    "display_width_px": tool.get("display_width_px"),
                    "display_height_px": tool.get("display_height_px"),
                }
                display_number = tool.get("display_number")
                if display_number is not None:
                    params["display_number"] = display_number

                computer_payload: dict[str, Any] = dict(tool)
                computer_payload["function"] = {
                    "name": name or "computer",
                    "parameters": params,
                }

                prepared.append(computer_payload)
                continue

            if self._transport_provider == "anthropic":
                prepared.append(dict(tool))
                continue

            prepared.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": input_schema,
                    },
                    "name": name,
                    "description": description,
                    "input_schema": input_schema,
                }
            )

        return prepared

    def _resolve_transport_layer(self) -> None:
        self._litellm_model = self.model
        self._resolved_completion_kwargs = dict(self.completion_kwargs)
        self._transport_provider = None

        if self.model.startswith("openrouter/"):
            self._transport_provider = "openrouter"
            self._litellm_model = self.model
        elif self.model.startswith("anthropic/"):
            mapped = ANTHROPIC_MODEL_ALIASES.get(self.model, self.model)
            self._litellm_model = mapped
            self._transport_provider = "anthropic"
        else:
            # Assume bare model names refer to Anthropic catalog for parity with claude.py
            # (LiteLLM expects provider prefixes for routing.)
            self._transport_provider = "anthropic"
            mapped = ANTHROPIC_MODEL_ALIASES.get(self.model, self.model)
            if not mapped.startswith("anthropic/"):
                mapped = f"anthropic/{mapped}"
            self._litellm_model = mapped
            self._resolved_completion_kwargs.pop("custom_llm_provider", None)

        if self._transport_provider == "anthropic":
            anthro_key = self._resolved_completion_kwargs.get("api_key") or settings.anthropic_api_key
            if not anthro_key:
                raise ValueError(
                    "Anthropic API key not found. Set ANTHROPIC_API_KEY or include 'api_key' in completion_kwargs."
                )
            self._resolved_completion_kwargs["api_key"] = anthro_key
            mapped_model = ANTHROPIC_MODEL_ALIASES.get(self._litellm_model, self._litellm_model)
            if not mapped_model.startswith("anthropic/"):
                mapped_model = f"anthropic/{mapped_model}"
            self._litellm_model = mapped_model
            self._resolved_completion_kwargs.pop("custom_llm_provider", None)

            self._ensure_anthropic_key_valid(anthro_key)

    def _set_computer_tool_type(self, new_type: str) -> None:
        if self._computer_tool_type == new_type:
            return
        updated = False
        for tool in self.claude_tools:
            if isinstance(tool.get("type"), str) and tool["type"].startswith("computer_"):
                tool["type"] = new_type
                updated = True
        if updated:
            self.hud_console.warning(
                f"Switching Anthropic computer tool type to {new_type} for LiteLLM compatibility"
            )
        self._computer_tool_type = new_type

    def _maybe_adjust_computer_tool(self, error: Exception) -> bool:
        message = str(error)
        if (
            self._computer_tool_type == "computer_20250124"
            and "computer_20250124" in message
            and "does not match any of the expected tags" in message
        ):
            self._set_computer_tool_type("computer_20241022")
            return True
        if (
            self._computer_tool_type == "computer_20241022"
            and "computer_20241022" in message
            and ("computer_20250124" in message or "Did you mean" in message)
        ):
            self._set_computer_tool_type("computer_20250124")
            return True
        return False

    def _ensure_anthropic_key_valid(self, api_key: str) -> None:
        if not self._validate_api_key or self._anthropic_key_validated:
            return

        try:
            from anthropic import Anthropic

            Anthropic(api_key=api_key).models.list()
            self._anthropic_key_validated = True
        except Exception as exc:  # pragma: no cover - Anthropic client validation
            raise ValueError(f"Anthropic API key is invalid: {exc}") from exc

    def _add_prompt_caching(self, messages: list[BetaMessageParam]) -> list[BetaMessageParam]:
        messages_cached = copy.deepcopy(messages)
        targets = set(self._cache_control_injection)

        if "system" in targets:
            for msg in messages_cached:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    self._apply_cache_control(msg)

        if "all_user" in targets:
            for msg in messages_cached:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    self._apply_cache_control(msg)
        elif "first_user" in targets:
            for msg in messages_cached:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    self._apply_cache_control(msg)
                    break

        if "last_user" in targets:
            if (
                messages_cached
                and isinstance(messages_cached[-1], dict)
                and messages_cached[-1].get("role") == "user"
            ):
                self._apply_cache_control(messages_cached[-1])

        return messages_cached

    def _apply_cache_control(self, message: dict[str, Any]) -> None:
        cache_value: BetaCacheControlEphemeralParam = {"type": "ephemeral"}
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"text", "image"}:
                    block["cache_control"] = cache_value
        elif isinstance(content, str) and content:
            message["cache_control"] = cache_value


def _safe_json_loads(raw: str | None) -> Any:
    if not raw:
        return {}
    return json.loads(raw)


def _extract_message_and_finish_reason(response: Any) -> tuple[Any, str | None]:
    if hasattr(response, "choices"):
        choices = getattr(response, "choices")
        if choices:
            choice = choices[0]
            return _get_field(choice, "message", {}), _get_field(choice, "finish_reason")
    if isinstance(response, dict) and response.get("choices"):
        choice = response["choices"][0]
        return choice.get("message", {}), choice.get("finish_reason")
    if isinstance(response, dict):
        return response, response.get("stop_reason")
    return response, None


def _get_field(obj: Any, key: str, default: Any | None = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def _to_json_safe(value: Any) -> Any:
    """Ensure tool inputs are JSON-serialisable for MCP calls."""

    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if _PydanticFieldInfo is not None and isinstance(value, _PydanticFieldInfo):
        default = getattr(value, "default", None)
        if isinstance(default, PydanticUndefined):
            default = None
        if default is not None:
            return _to_json_safe(default)
        default_factory = getattr(value, "default_factory", None)
        if callable(default_factory):  # pragma: no cover - rare path
            try:
                produced = default_factory()
            except Exception:
                produced = None
            return _to_json_safe(produced)
        example = getattr(value, "examples", None)
        if example:
            return _to_json_safe(example[0])
        extra = getattr(value, "json_schema_extra", None)
        if isinstance(extra, dict):
            hint = extra.get("example") or extra.get("examples")
            if hint:
                if isinstance(hint, list):
                    return _to_json_safe(hint[0])
                return _to_json_safe(hint)
        return None
    return value


def base64_to_content_block(base64: str) -> BetaImageBlockParam:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64},
    }


def text_to_content_block(text: str) -> BetaTextBlockParam:
    return {"type": "text", "text": text}


def tool_use_content_block(
    tool_use_id: str, content: list[BetaTextBlockParam | BetaImageBlockParam]
) -> BetaToolResultBlockParam:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
