import asyncio
import json
import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any, cast

from anthropic import APIStatusError
from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
)
from mcp.types import ImageContent, TextContent

from .agent import BaseAgent
from .messages_to_raw_prompt import anthropic_messages_to_raw_prompt
from .messages_to_raw_prompt import MessageParam as PromptMessageParam

logger = logging.getLogger(__name__)


# Hardcoded allowlist of Anthropic models (short ids)
ANTHROPIC_ALLOWED_MODELS: list[str] = [
    "claude-3-7-sonnet-20250219",
    "claude-4-sonnet-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3.5-haiku",
    "claude-3-opus",
    "claude-3-haiku",
]

# Model handling is now consistent across providers - no complex mapping needed


def truncate_base64_in_data(data, max_length=100):
    """Recursively truncate base64 strings in data structures for logging."""
    import re
    
    if isinstance(data, dict):
        return {k: truncate_base64_in_data(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_base64_in_data(item, max_length) for item in data]
    elif isinstance(data, str):
        # Pattern to match base64 strings (common patterns used in image data)
        base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/]+=*|[A-Za-z0-9+/]{50,}=*'
        
        def replace_base64(match):
            b64_str = match.group(0)
            if len(b64_str) > max_length:
                return f"[BASE64_DATA_TRUNCATED_{len(b64_str)}_CHARS]"
            return b64_str
            
        return re.sub(base64_pattern, replace_base64, data)
    else:
        return data


def mcp_content_block_to_messages_format(content: TextContent | ImageContent):
    if isinstance(content, TextContent):
        return TextBlockParam(type="text", text=content.text)
    elif isinstance(content, ImageContent):
        assert content.mimeType in ("image/jpeg", "image/png", "image/gif", "image/webp")
        return ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                data=content.data,
                media_type=content.mimeType,
                type="base64",
            ),
        )
    else:
        raise ValueError(f"Invalid content type: {type(content)}")


class MessageConverter(ABC):
    """Abstract base class for converting messages between different provider formats."""

    @abstractmethod
    def convert_messages_to_provider_format(self, messages: list[dict[str, Any]]) -> Any:
        """Convert internal message format to provider-specific format."""
        pass

    @abstractmethod
    def convert_tools_to_provider_format(self, tools: list[ToolParam]) -> Any:
        """Convert internal tool format to provider-specific format."""
        pass


class AnthropicMessageConverter(MessageConverter):
    """Converts messages for Anthropic API."""

    def convert_messages_to_provider_format(self, messages: list[dict[str, Any]]) -> list[MessageParam]:
        """Convert internal messages to Anthropic format."""
        return cast(list[MessageParam], messages)

    def convert_tools_to_provider_format(self, tools: list[ToolParam]) -> list[ToolParam]:
        """Convert internal tools to Anthropic format.

        - Ensure each tool is a plain dict with only name, description, input_schema
        - Ensure input_schema has top-level type: object
        - Strip any non-spec keys (e.g., cache_control accidentally added upstream)
        """
        sanitized: list[ToolParam] = []
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name", "")
                description = tool.get("description", "")
                input_schema = tool.get("input_schema", {}) or {}
                # Coerce input_schema to a dict and ensure type is object
                if not isinstance(input_schema, dict):
                    input_schema = {}
                if "type" not in input_schema:
                    input_schema = {"type": "object", **input_schema}
                # Only keep allowed keys
                sanitized.append({
                    "name": name,
                    "description": description,
                    "input_schema": input_schema,
                })
            else:
                # For SDK ToolParam instances, pass through
                sanitized.append(tool)
        return sanitized


class OpenAIMessageConverter(MessageConverter):
    """Placeholder kept for import stability; not used in ClaudeAgent."""
    def convert_messages_to_provider_format(self, messages: list[dict[str, Any]]) -> Any:  # pragma: no cover
        return messages
    def convert_tools_to_provider_format(self, tools: list[ToolParam]) -> Any:  # pragma: no cover
        return tools


class ToolCallHandler(ABC):
    """Abstract base class for handling tool calls from different providers."""

    @abstractmethod
    async def process_stream_chunk(self, chunk: Any) -> dict[str, Any] | None:
        """Process a single stream chunk and return unified events."""
        pass

    @abstractmethod
    def finalize_tool_calls(self, tool_calls: list[dict[str, Any]], current_tool_use: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Finalize any pending tool calls."""
        pass


class AnthropicToolCallHandler(ToolCallHandler):
    """Handles tool calls from Anthropic API."""

    def __init__(self):
        self.current_partial_json = ""
        self.current_tool_use: dict[str, Any] | None = None

    async def process_stream_chunk(self, chunk: Any) -> dict[str, Any] | None:
        """Process Anthropic stream chunk."""
        if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
            return {"type": "text", "content": chunk.delta.text}

        elif chunk.type == "content_block_start" and chunk.content_block.type == "tool_use":
            assert isinstance(chunk, RawContentBlockStartEvent)
            self.current_tool_use = {"id": chunk.content_block.id, "name": chunk.content_block.name, "input": {}, "error": None}
            return {"type": "tool_use_start", "tool": chunk.content_block.name}

        elif chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
            assert isinstance(chunk, RawContentBlockDeltaEvent)
            if chunk.delta.partial_json:
                if not self.current_tool_use:
                    logger.warning(f"No current tool use found for JSON delta: {chunk.delta.partial_json}")
                    return None
                self.current_partial_json += chunk.delta.partial_json
                return {"type": "tool_use_param", "value": chunk.delta.partial_json}

        elif chunk.type == "content_block_stop" and self.current_tool_use:
            assert isinstance(chunk, RawContentBlockStopEvent)
            # Try to parse the JSON
            try:
                parsed_partial_json = json.loads(self.current_partial_json)
                self.current_tool_use["input"] = parsed_partial_json
            except json.JSONDecodeError as e:
                self.current_tool_use["error"] = f"Unable to parse JSON: {str(e)}. Raw JSON:\n```\n{self.current_partial_json}\n```"
            # Reset for next tool call
            tool_call = self.current_tool_use
            self.current_tool_use = None
            self.current_partial_json = ""
            return {"type": "tool_call_complete", "tool_call": tool_call}

        # Handle max_tokens stop to propagate partial tool call error like normal flow
        elif getattr(chunk, "type", None) == "message_delta" and getattr(getattr(chunk, "delta", None), "stop_reason", None) == "max_tokens":
            if self.current_tool_use:
                self.current_tool_use["error"] = f"Response exceeded max_tokens during stream."
                tool_call = self.current_tool_use
                self.current_tool_use = None
                self.current_partial_json = ""
                return {"type": "tool_call_complete", "tool_call": tool_call}

        return None

    def finalize_tool_calls(self, tool_calls: list[dict[str, Any]], current_tool_use: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Anthropic doesn't need finalization since it handles completion in process_stream_chunk."""
        return tool_calls


class OpenRouterToolCallHandler(ToolCallHandler):
    """Placeholder; not used in ClaudeAgent."""
    def __init__(self):
        pass
    async def process_stream_chunk(self, chunk: Any) -> dict[str, Any] | None:  # pragma: no cover
        return None
    def finalize_tool_calls(self, tool_calls: list[dict[str, Any]], current_tool_use: dict[str, Any] | None) -> list[dict[str, Any]]:  # pragma: no cover
        return tool_calls


class ProviderAdapter(ABC):
    """Abstract base class for provider-specific adapters."""

    def __init__(self, message_converter: MessageConverter, tool_handler: ToolCallHandler):
        self.message_converter = message_converter
        self.tool_handler = tool_handler

    @abstractmethod
    async def create_client(self, **kwargs) -> Any:
        """Create and return a provider-specific client."""
        pass

    @abstractmethod
    async def make_request(self, client: Any, messages: list[dict[str, Any]], tools: list[ToolParam], **kwargs) -> Any:
        """Make a request to the provider."""
        pass


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API."""

    def __init__(self, auth_mode: str = "normal"):
        super().__init__(AnthropicMessageConverter(), AnthropicToolCallHandler())
        self.auth_mode = auth_mode

    async def create_client(self, **kwargs) -> Any:
        """Anthropic client is passed in via kwargs."""
        return kwargs.get('anthropic_client')

    async def make_request(self, client: Any, messages: list[dict[str, Any]], tools: list[ToolParam], **kwargs) -> Any:
        """Make request to Anthropic API."""
        extra_headers = {
            "anthropic-beta": "computer-use-2025-01-24,fine-grained-tool-streaming-2025-05-14",
        }
        system_prompt = None

        # OAuth mode removed; always use normal headers
        params: dict[str, Any] = {
            "model": kwargs["model"],
            "max_tokens": kwargs["max_tokens"],
            "messages": self.message_converter.convert_messages_to_provider_format(messages),
            "tools": self.message_converter.convert_tools_to_provider_format(tools),
            "extra_headers": extra_headers,
            "stream": True,
        }
        # Only include system if provided; sending None can trigger validation errors
        if system_prompt is not None:
            params["system"] = system_prompt

        return await client.messages.create(**params)


class OpenRouterAdapter(ProviderAdapter):
    """Placeholder; not used in ClaudeAgent."""
    def __init__(self):
        super().__init__(OpenAIMessageConverter(), OpenRouterToolCallHandler())
    async def create_client(self, **kwargs) -> Any:  # pragma: no cover
        return None
    async def make_request(self, client: Any, messages: list[dict[str, Any]], tools: list[ToolParam], **kwargs) -> Any:  # pragma: no cover
        raise NotImplementedError


class ClaudeAgent(BaseAgent):
    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        from .main import MCPClient

        # Create MCP client and get the problem prompt
        mcp_client = MCPClient(self.problem, self.image_override, runtime=self.runtime, delete_mode=self.delete_mode)

        try:
            logger.info("Connecting to server")
            await mcp_client.connect_to_server()
            logger.info("Connected to server")

            # Get available tools first
            tools = cast(list[ToolParam], await mcp_client.get_tools())
            if not tools:
                raise RuntimeError("No tools detected from MCP server")
            tools[-1]["cache_control"] = {"type": "ephemeral"}

            # Emit start event early with container info & port mappings
            yield {
                "type": "start",
                "tools": tools,
                "container_name": mcp_client.container_name,
                "ports": mcp_client.ports,
            }

            # Proceed to obtain prompt (may take longer)
            prompt = await mcp_client.get_prompt()
            logger.info(f"Prompt: {truncate_base64_in_data(prompt)}")
            logger.info(f"Max LLM steps configured: {self.max_llm_steps}")
            tools = cast(list[ToolParam], await mcp_client.get_tools())
            logger.info(f"Tools: {tools}")
            if not tools:
                raise RuntimeError("No tools detected from MCP server")
            tools[-1]["cache_control"] = {"type": "ephemeral"}

            yield {
                "type": "start",
                "tools": tools,
                "container_name": mcp_client.container_name,
                "ports": mcp_client.ports,
            }
        except Exception as e:
            logger.exception("Error setting up container")
            error_details = traceback.format_exc()
            yield {"type": "error", "content": f"Error setting up container:\n{str(e)}\n\nStack trace:\n{error_details}"}
            return

        assert isinstance(prompt, list)
        assert len(prompt) == 1, "Only one text block is supported"
        assert isinstance(prompt[0], TextContent), f"Expected TextBlock, got {type(prompt[0])}"
        assert isinstance(prompt[0].text, str)

        first_message = {
            "role": "user",
            "content": [{"type": "text", "text": self.custom_prompt or prompt[0].text}],
        }
        messages: list[dict[str, Any]] = [first_message]
        yield {"type": "message", "message": first_message}

        try:
            llm_step_count = 0
            while True:
                logger.info(f"#### MODEL TURN {len(messages)} ####")
                logger.info(f"Current LLM step count: {llm_step_count}, Max steps: {self.max_llm_steps}")
                has_tool_call = False

                messages_clone = deepcopy(messages)

                # Mark last user message with cache control for prompt caching
                last_msg = messages_clone[-1]
                if last_msg.get("role") == "user":
                    for block in last_msg.get("content", []):
                        block["cache_control"] = {"type": "ephemeral"}

                retries = 0
                max_retries = 10
                base_delay = 1
                while True:
                    try:
                        logger.info(f"Tools: {tools}")
                        # Always use Anthropic SDK client directly here
                        extra_headers: dict[str, str] = {
                            "anthropic-beta": "computer-use-2025-01-24,fine-grained-tool-streaming-2025-05-14",
                        }

                        response = await mcp_client.anthropic.messages.create(
                            model=self.model,
                            max_tokens=self.max_tokens,
                            messages=cast(list[MessageParam], messages_clone),
                            tools=tools,
                            extra_headers=extra_headers,
                            stream=True,
                        )
                        break
                    except Exception as e:
                        logger.exception(f"API call failed: {e}")
                        if retries >= max_retries:
                            raise
                        retries += 1
                        delay = base_delay * (2 ** (retries - 1))
                        logger.warning(f"API call failed, attempt {retries}/{max_retries}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)

                logger.info(f"#### MODEL TURN {len(messages)} RESPONSE ####")

                tool_calls: list[dict[str, Any]] = []
                current_partial_json = ""
                current_text = ""
                current_tool_use: dict[str, Any] | None = None

                # Process the stream chunks for this model turn
                async for chunk in response:
                    if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        current_text += chunk.delta.text
                        yield {"type": "text", "content": chunk.delta.text}

                    elif chunk.type == "content_block_start" and chunk.content_block.type == "tool_use":
                        assert isinstance(chunk, RawContentBlockStartEvent)
                        has_tool_call = True
                        current_tool_use = {"id": chunk.content_block.id, "name": chunk.content_block.name, "input": {}, "error": None}
                        yield {"type": "tool_use_start", "tool": chunk.content_block.name}

                    elif chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
                        assert isinstance(chunk, RawContentBlockDeltaEvent)
                        if chunk.delta.partial_json:
                            if not current_tool_use:
                                logger.warning(f"No current tool use found for JSON delta: {chunk.delta.partial_json}")
                                continue
                            current_partial_json += chunk.delta.partial_json
                            yield {"type": "tool_use_param", "value": chunk.delta.partial_json}

                    elif chunk.type == "content_block_stop" and current_tool_use:
                        assert isinstance(chunk, RawContentBlockStopEvent)
                        # Try to parse the JSON for the tool call
                        try:
                            parsed_partial_json = json.loads(current_partial_json)
                            current_tool_use["input"] = parsed_partial_json
                        except json.JSONDecodeError as e:
                            current_tool_use["error"] = f"Unable to parse JSON: {str(e)}. Raw JSON:\n```\n{current_partial_json}\n```"

                        tool_calls.append(current_tool_use)
                        current_tool_use = None

                    elif getattr(chunk, "type", None) == "message_delta" and getattr(getattr(chunk, "delta", None), "stop_reason", None) == "max_tokens":
                        if current_tool_use:
                            current_tool_use["error"] = f"Response exceeded max_tokens during stream."
                            tool_calls.append(current_tool_use)
                            current_tool_use = None

                for tool_call in tool_calls:
                    if tool_call["input"] == {}:
                        logger.warning(f"Tool call {tool_call['name']} has no input")
                        logger.warning(f"Tool call details: {tool_call}")

                assistant_content: list[dict[str, Any]] = []
                if current_text:
                    assistant_content.append({"type": "text", "text": current_text})
                for tool_call in tool_calls:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "input": tool_call["input"],
                        }
                    )

                assistant_message = {"role": "assistant", "content": assistant_content}
                if len(assistant_message["content"]) == 0:
                    assistant_message["content"] = [{"type": "text", "text": "No content generated"}]
                messages.append(assistant_message)
                yield {"type": "message", "message": assistant_message}

                # Increment LLM step count and check if we've hit the limit
                llm_step_count += 1
                if self.max_llm_steps > 0 and llm_step_count >= self.max_llm_steps:
                    logger.info(f"Reached max LLM steps ({self.max_llm_steps}), terminating interaction")
                    break

                if has_tool_call:
                    tool_results: list[dict[str, Any]] = []
                    logger.info(f"Detected tool calls: {tool_calls}")
                    for tool_call in tool_calls:
                        yield {"type": "tool_executing", "tool": tool_call["name"], "args": tool_call["input"]}
                        try:
                            if tool_call["error"] is not None:
                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "is_error": True,
                                        "tool_use_id": tool_call["id"],
                                        "content": [{"type": "text", "text": tool_call["error"]}],
                                    }
                                )
                            else:
                                logger.info(f"#### >> BEGIN TOOL CALL {tool_call['name']} ####")
                                logger.info(truncate_base64_in_data(tool_call["input"]))
                                tool_result = await mcp_client.execute_tool(tool_call["name"], tool_call["input"])
                                logger.info(f"#### << END TOOL CALL {tool_call['name']} ####")
                                tool_results.append(
                                    {
                                        "type": "tool_result",
                                        "is_error": tool_result.isError,
                                        "tool_use_id": tool_call["id"],
                                        "content": [
                                            mcp_content_block_to_messages_format(content)
                                            for content in tool_result.content
                                            if isinstance(content, (TextContent, ImageContent))
                                        ],
                                    }
                                )
                        except Exception as e:
                            error_details = traceback.format_exc()
                            error_message = f"Tool execution error:\n{str(e)}\n\nStack trace:\n{error_details}"
                            yield {"type": "error", "content": error_message}

                    tool_result_message = {"role": "user", "content": tool_results}
                    if len(tool_results) > 0:
                        messages.append(tool_result_message)
                    else:
                        logger.warning(f"tool calls: {tool_calls}")
                        tool_result_message = {"role": "user", "content": [{"type": "text", "text": "No tool results"}]}
                        messages.append(tool_result_message)
                    yield {"type": "message", "message": tool_result_message}
                    continue
                else:
                    break

            raw_prompt = anthropic_messages_to_raw_prompt(
                "[system prompt placeholder]\n\n", messages
            )
            async for evt in self._grade_and_emit(mcp_client, raw_prompt):
                yield evt
        except Exception as e:
            logger.exception(f"Error during stream: {e}")
            error_details = traceback.format_exc()
            yield {"type": "error", "content": f"Stream error:\n{str(e)}\n\nStack trace:\n{error_details}"}
        finally:
            await mcp_client.cleanup()

    # No OpenAI conversion logic in Anthropic-only agent
