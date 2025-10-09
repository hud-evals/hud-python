"""OpenRouter agent that uses the Responses API with prompt caching."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Iterable

import mcp.types as types
from openai import AsyncOpenAI

from hud import instrument
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult

from .openai_chat_generic import GenericOpenAIChatAgent

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/alpha"
_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://hud.so",
    "X-Title": "HUD Python SDK",
    "Accept": "application/json",
}

_DEFAULT_COMPLETION_KWARGS: dict[str, Any] = {
    "temperature": 0.1,
    "max_output_tokens": 1024,
}


class OpenRouterAgent(GenericOpenAIChatAgent):
    """MCP-enabled agent that talks to OpenRouter through the Responses API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str = "z-ai/glm-4.5v",
        default_headers: dict[str, str] | None = None,
        cache_control: dict[str, Any] | bool | None = True,
        cacheable_roles: Iterable[str] | None = None,
        openai_client: AsyncOpenAI | None = None,
        completion_kwargs: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        api_key = api_key or settings.openrouter_api_key
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY or pass api_key explicitly."
            )

        base_url = base_url or _DEFAULT_BASE_URL

        headers: dict[str, str] = dict(_DEFAULT_HEADERS)
        if default_headers:
            headers.update(default_headers)

        client = openai_client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers,
        )

        super().__init__(
            openai_client=client,
            model_name=model_name,
            completion_kwargs=completion_kwargs,
            **agent_kwargs,
        )

        self._responses_kwargs = {
            "tool_choice": "auto",
            **_DEFAULT_COMPLETION_KWARGS,
            **dict(self.completion_kwargs),
        }
        self.completion_kwargs.clear()

        self._cache_control = self._normalize_cache_control(cache_control)
        self._cacheable_roles = tuple(cacheable_roles or ("system", "user", "tool"))

    @staticmethod
    def _normalize_cache_control(
        cache_control: dict[str, Any] | bool | str | None,
    ) -> dict[str, Any] | None:
        if cache_control is False:
            return None
        if cache_control is None:
            return {"type": "ephemeral"}
        if cache_control is True:
            return {"type": "ephemeral"}
        if isinstance(cache_control, dict):
            return cache_control
        return {"type": str(cache_control)}

    def _should_cache(self, role: str) -> bool:
        return self._cache_control is not None and role in self._cacheable_roles

    def _text_item(self, text: str, role: str) -> dict[str, Any]:
        item: dict[str, Any] = {"type": "input_text", "text": text}
        if self._should_cache(role):
            item["cache_control"] = self._cache_control
        return item

    def _image_item(self, image_payload: Any, role: str) -> dict[str, Any]:
        url: str | None = None
        detail = None

        if isinstance(image_payload, dict):
            # Standard OpenAI-style wrapper
            if "image_url" in image_payload and isinstance(image_payload["image_url"], dict):
                img = image_payload["image_url"]
                url = img.get("url")
                detail = img.get("detail") or image_payload.get("detail")
            # Direct url / data uri
            elif image_payload.get("url"):
                url = image_payload.get("url")
                detail = image_payload.get("detail")
            # Raw base64 payload from computer/tool results
            elif image_payload.get("data"):
                mime = (
                    image_payload.get("mimeType")
                    or image_payload.get("mime_type")
                    or "image/png"
                )
                data = image_payload.get("data")
                if data:
                    url = f"data:{mime};base64,{data}"
                detail = image_payload.get("detail")
            elif isinstance(image_payload.get("source"), dict):
                source = image_payload["source"]
                data = source.get("data")
                mime = source.get("media_type") or source.get("mime_type") or "image/png"
                if data:
                    url = f"data:{mime};base64,{data}"
                detail = source.get("detail")
        elif isinstance(image_payload, str):
            url = image_payload

        item: dict[str, Any] = {"type": "input_image"}
        if url:
            item["image_url"] = url
        item["detail"] = str(detail or "auto")
        if self._should_cache(role):
            item["cache_control"] = self._cache_control
        return item

    def _convert_message_content(self, role: str, content: Any) -> list[dict[str, Any]]:
        if content is None:
            return []

        blocks: list[dict[str, Any]] = []
        if isinstance(content, str):
            blocks.append(self._text_item(content, role))
            return blocks

        if isinstance(content, dict):
            content = [content]

        if isinstance(content, list):
            for entry in content:
                if isinstance(entry, str):
                    blocks.append(self._text_item(entry, role))
                elif isinstance(entry, dict):
                    entry_copy = dict(entry)
                    entry_type = entry_copy.get("type")
                    if entry_type in {"text", "input_text", None}:
                        text = entry_copy.get("text") or ""
                        blocks.append(self._text_item(text, role))
                    elif entry_type in {"image_url", "input_image"}:
                        payload = entry_copy.get("image_url", entry_copy.get("image")) or entry_copy
                        blocks.append(self._image_item(payload, role))
                    elif entry_type in {"image", "output_image", "rendered"}:
                        blocks.append(self._image_item(entry_copy, role))
                    elif entry_type == "tool_result":
                        text = entry_copy.get("text", "")
                        blocks.append(self._text_item(text, role))
                    else:
                        text_value = entry_copy.get("text") or json.dumps(entry_copy)
                        blocks.append(self._text_item(text_value, role))
                else:
                    blocks.append(self._text_item(str(entry), role))
            return blocks

        blocks.append(self._text_item(str(content), role))
        return blocks

    def _convert_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                logger.debug("Skipping non-dict message: %s", message)
                continue

            if "type" in message and "role" not in message:
                converted.append(message)
                continue

            role = message.get("role") or "user"

            if role == "assistant" and message.get("tool_calls"):
                content_items = self._convert_message_content(role, message.get("content"))
                if content_items:
                    converted.append({"role": "assistant", "content": content_items})
                for tool_call in message.get("tool_calls", []):
                    converted.append(self._convert_tool_call(tool_call))
                continue

            if role == "tool":
                converted.extend(self._convert_tool_message(message))
                continue

            payload: dict[str, Any] = {"role": role}
            content_items = self._convert_message_content(role, message.get("content"))
            if content_items:
                payload["content"] = content_items
            if message.get("name"):
                payload["name"] = message["name"]
            if message.get("metadata"):
                payload["metadata"] = message["metadata"]
            converted.append(payload)

        return converted

    @staticmethod
    def _jsonify_schema(value: Any) -> Any:
        from pydantic import BaseModel
        from pydantic.fields import FieldInfo

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        if isinstance(value, dict):
            return {str(k): OpenRouterAgent._jsonify_schema(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [OpenRouterAgent._jsonify_schema(v) for v in value]

        try:
            return json.loads(json.dumps(value))
        except Exception:
            if isinstance(value, BaseModel):
                return OpenRouterAgent._jsonify_schema(value.model_dump())
            if isinstance(value, FieldInfo):
                data: dict[str, Any] = {}
                if value.annotation is not None:
                    data.setdefault(
                        "type",
                        getattr(value.annotation, "__name__", str(value.annotation)),
                    )
                if value.description:
                    data["description"] = value.description
                if value.title:
                    data["title"] = value.title
                if value.default not in (None, Ellipsis):
                    data["default"] = OpenRouterAgent._jsonify_schema(value.default)
                if value.json_schema_extra:
                    extra = OpenRouterAgent._jsonify_schema(value.json_schema_extra)
                    if isinstance(extra, dict):
                        data.update(extra)
                return data or str(value)
            if hasattr(value, "model_dump"):
                return OpenRouterAgent._jsonify_schema(value.model_dump())
            if hasattr(value, "__dict__") and value.__dict__:
                return OpenRouterAgent._jsonify_schema(
                    {
                        k: v
                        for k, v in value.__dict__.items()
                        if not k.startswith("_")
                    }
                )
            return str(value)

    @staticmethod
    def _convert_tools_for_responses(tools: list[dict] | None) -> list[dict]:
        if not tools:
            return []

        converted: list[dict] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue

            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                fn = tool["function"]
                name = fn.get("name")
                params = fn.get("parameters", {})
                description = fn.get("description", "")

                if not isinstance(name, str) or not name:
                    logger.debug("Skipping tool with missing name: %s", tool)
                    continue

                converted.append(
                    {
                        "type": "function",
                        "name": name,
                        "description": str(description or ""),
                        "parameters": OpenRouterAgent._jsonify_schema(params),
                    }
                )
            else:
                converted.append(OpenRouterAgent._jsonify_schema(tool))

        return converted

    def _convert_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tool_call, dict):
            return {}

        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or "tool_call"
        raw_arguments = function.get("arguments")

        if isinstance(raw_arguments, dict):
            arguments = json.dumps(self._jsonify_schema(raw_arguments))
        elif isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = raw_arguments
            else:
                arguments = json.dumps(self._jsonify_schema(parsed))
        elif raw_arguments is None:
            arguments = "{}"
        else:
            arguments = json.dumps(self._jsonify_schema(raw_arguments))

        call_id = (
            tool_call.get("id")
            or function.get("id")
            or function.get("call_id")
            or f"call_{uuid.uuid4().hex}"
        )

        return {
            "type": "function_call",
            "id": call_id,
            "name": name,
            "arguments": arguments or "{}",
        }

    def _convert_tool_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        call_id = message.get("tool_call_id") or message.get("id") or f"call_{uuid.uuid4().hex}"

        text_parts: list[str] = []
        image_payloads: list[Any] = []

        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {"text", "input_text"} and item.get("text"):
                        text_parts.append(str(item.get("text")))
                    elif item_type in {"image", "input_image", "image_url", "output_image", "rendered"}:
                        image_payloads.append(item)
                elif isinstance(item, str):
                    text_parts.append(item)
        elif isinstance(content, str):
            text_parts.append(content)

        structured = message.get("structuredContent")
        if structured and not text_parts:
            try:
                text_parts.append(json.dumps(structured))
            except Exception:
                text_parts.append(str(structured))

        output_text = "\n".join(part for part in text_parts if part) or ""

        entries.append(
            {
                "type": "function_call_output",
                "id": message.get("id") or call_id,
                "call_id": call_id,
                "output": output_text,
            }
        )

        for payload in image_payloads:
            entries.append(
                {
                    "role": "user",
                    "content": [self._image_item(payload, "user")],
                }
            )

        return entries

    async def format_tool_results(
        self,
        tool_calls: list[MCPToolCall],
        tool_results: list[MCPToolResult],
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []

        for call, result in zip(tool_calls, tool_results, strict=False):
            call_id = call.id or call.name or f"call_{uuid.uuid4().hex}"

            text_parts: list[str] = []
            image_payloads: list[Any] = []

            for item in result.content or []:
                if isinstance(item, types.TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, types.ImageContent):
                    image_payloads.append(
                        {
                            "mimeType": item.mimeType,
                            "data": item.data,
                            "detail": getattr(item, "detail", None),
                        }
                    )
                elif isinstance(item, dict):
                    if item.get("type") in {"text", "input_text"}:
                        text_parts.append(str(item.get("text", "")))
                    elif item.get("type") in {"image", "input_image", "image_url", "output_image", "rendered"}:
                        image_payloads.append(item)
                elif isinstance(item, str):
                    text_parts.append(item)

            if result.structuredContent and not text_parts:
                try:
                    text_parts.append(json.dumps(result.structuredContent))
                except Exception:
                    text_parts.append(str(result.structuredContent))

            if getattr(result, "isError", False):
                text_parts.append(getattr(result, "error", "Tool execution failed."))

            output_text = "\n".join(part for part in text_parts if part) or ""

            converted.append(
                {
                    "type": "function_call_output",
                    "id": call_id,
                    "call_id": call_id,
                    "output": output_text,
                }
            )

            for payload in image_payloads:
                converted.append(
                    {
                        "role": "user",
                        "content": [self._image_item(payload, "user")],
                    }
                )

        return converted

    @staticmethod
    def _parse_arguments(arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments:
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                logger.debug("Failed to decode arguments: %s", arguments)
        return {}

    def _to_mcp_tool_call(self, payload: dict[str, Any]) -> MCPToolCall:
        tool_name = payload.get("name") or payload.get("function", {}).get("name") or ""
        call_id = payload.get("id") or payload.get("tool_call_id") or payload.get("call_id")
        if not call_id:
            call_id = tool_name
        arguments = payload.get("arguments")
        if not arguments and "function" in payload:
            arguments = payload["function"].get("arguments")
        parsed_arguments = self._parse_arguments(arguments)
        return MCPToolCall(id=call_id, name=tool_name, arguments=parsed_arguments)

    def _coerce_response_payload(self, response: Any) -> dict[str, Any]:
        """Convert OpenRouter SDK return types into a plain dictionary."""

        if response is None:
            return {}

        if isinstance(response, dict):
            return response

        for attr in ("model_dump", "dict", "to_dict"):
            if hasattr(response, attr):
                try:
                    payload = getattr(response, attr)()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Failed to read response via %s: %s", attr, exc)
                else:
                    if isinstance(payload, dict):
                        return payload

        snapshot = getattr(response, "__dict__", None)
        if isinstance(snapshot, dict):
            return snapshot

        logger.error("Unexpected response carrier from OpenRouter: %r", response)
        raise TypeError("Unexpected response type from OpenRouter")

    def _extract_response(self, response: Any) -> AgentResponse:
        data = self._coerce_response_payload(response)
        if not isinstance(data, dict):
            raise TypeError("Unexpected response type from OpenRouter")

        output = data.get("output", [])
        text_parts: list[str] = []
        tool_calls: list[MCPToolCall] = []
        reasoning_parts: list[str] = []

        for item in output:
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "message":
                contents = item.get("content", [])
                if isinstance(contents, list):
                    for block in contents:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        if block_type in {"output_text", "text"}:
                            text = block.get("text")
                            if text:
                                text_parts.append(text)
                        elif block_type == "reasoning" and block.get("text"):
                            reasoning_parts.append(block["text"])
                for tc in item.get("tool_calls", []) or []:
                    if isinstance(tc, dict):
                        tool_calls.append(self._to_mcp_tool_call(tc))
            elif item_type in {"tool_call", "function_call"} and isinstance(item, dict):
                tool_calls.append(self._to_mcp_tool_call(item))
            elif item_type == "reasoning" and isinstance(item, dict):
                summary = item.get("summary")
                if isinstance(summary, list):
                    for block in summary:
                        if isinstance(block, dict) and block.get("text"):
                            reasoning_parts.append(block["text"])
                elif isinstance(summary, str):
                    reasoning_parts.append(summary)

        merged_text = "\n".join(reasoning_parts + text_parts).strip()
        status = data.get("status", "completed")
        done = not tool_calls and status != "in_progress"
        return AgentResponse(
            content=merged_text,
            tool_calls=tool_calls,
            done=done,
            raw=response,
        )

    @instrument(
        span_type="agent",
        record_args=False,
        record_result=True,
    )
    async def get_response(self, messages: list[Any]) -> AgentResponse:
        converted_messages = self._convert_messages(messages)
        tools = self._convert_tools_for_responses(self.get_tool_schemas())

        protected_keys = {"model", "input", "tools"}
        extra = {k: v for k, v in self._responses_kwargs.items() if k not in protected_keys}
        # If tools are provided and tool_choice isn't explicitly set, require tool use
        if tools and "tool_choice" not in extra:
            extra["tool_choice"] = "required"

        try:
            payload: dict[str, Any] = {
                "model": self.model_name,
                "input": converted_messages,
                **extra,
            }
            if tools:
                payload["tools"] = tools

            response = await self.oai.responses.create(**payload)
        except Exception as exc:
            error_content = f"Error getting response {exc}"
            logger.exception("OpenRouter call failed: %s", exc)
            return AgentResponse(
                content=error_content,
                tool_calls=[],
                done=True,
                isError=True,
                raw=None,
            )

        return self._extract_response(response)
