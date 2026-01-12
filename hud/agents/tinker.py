"""Tinker Computer Use Agent implementation.

This agent uses Tinker's sampling API for model inference, designed for
reinforcement learning integration where prompts, tokens, and logprobs
need to be captured for training.

Key features:
- Stores completions with prompts, tokens, and logprobs for RL training
- Supports computer use with proper coordinate normalization
- Compatible with the HUD EvalContext for tool execution
- Can be used with inference-service for rollout generation
"""

from __future__ import annotations

import base64
import contextvars
import io
import json
import logging
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

import mcp.types as types
from PIL import Image
from pydantic import ConfigDict, Field
from tinker_cookbook.renderers.base import ToolCall

from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, BaseAgentConfig, MCPToolCall, MCPToolResult
from hud.utils.types import with_signature

from .base import BaseCreateParams, MCPAgent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Context var for tracking current episode ID (for RL training)
_CURRENT_EPISODE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tinker_current_episode_id", default=None
)


def set_current_episode_id(episode_id: str) -> contextvars.Token[str | None]:
    """Bind subsequent model samples (within this async context) to the given episode id."""
    return _CURRENT_EPISODE_ID.set(episode_id)


def reset_current_episode_id(token: contextvars.Token[str | None]) -> None:
    _CURRENT_EPISODE_ID.reset(token)


def get_current_episode_id() -> str | None:
    return _CURRENT_EPISODE_ID.get()


@dataclass(frozen=True)
class TinkerSampleRecord:
    """Record of a Tinker sample for RL training."""

    completion_id: str
    prompt: Any  # tinker.ModelInput
    tokens: list[int]
    logprobs: list[float] | None
    finish_reason: str
    created_at: float


class TinkerCompletionStore:
    """Thread-safe store for Tinker completions used in RL training."""

    def __init__(self) -> None:
        self._records: dict[str, TinkerSampleRecord] = {}
        self._order: deque[str] = deque()
        self._episodes: dict[str, deque[str]] = {}
        self._completion_episode: dict[str, str] = {}
        self._lock = Lock()

    def add(self, record: TinkerSampleRecord, *, episode_id: str | None = None) -> None:
        with self._lock:
            self._records[record.completion_id] = record
            self._order.append(record.completion_id)
            if episode_id:
                self._episodes.setdefault(episode_id, deque()).append(record.completion_id)
                self._completion_episode[record.completion_id] = episode_id

    def pop(self, completion_id: str) -> TinkerSampleRecord | None:
        with self._lock:
            record = self._records.pop(completion_id, None)
            if record:
                try:
                    self._order.remove(completion_id)
                except ValueError:
                    pass
                episode_id = self._completion_episode.pop(completion_id, None)
                if episode_id:
                    try:
                        self._episodes.get(episode_id, deque()).remove(completion_id)
                    except ValueError:
                        pass
                    if episode_id in self._episodes and not self._episodes[episode_id]:
                        self._episodes.pop(episode_id, None)
            return record

    def pop_many(self, completion_ids: Sequence[str]) -> list[TinkerSampleRecord]:
        popped: list[TinkerSampleRecord] = []
        with self._lock:
            for completion_id in completion_ids:
                record = self._records.pop(completion_id, None)
                if record:
                    try:
                        self._order.remove(completion_id)
                    except ValueError:
                        pass
                    episode_id = self._completion_episode.pop(completion_id, None)
                    if episode_id:
                        try:
                            self._episodes.get(episode_id, deque()).remove(completion_id)
                        except ValueError:
                            pass
                        if episode_id in self._episodes and not self._episodes[episode_id]:
                            self._episodes.pop(episode_id, None)
                    popped.append(record)
        return popped

    def pop_episode(self, episode_id: str) -> list[TinkerSampleRecord]:
        """Pop all records for a given episode in the order they were generated."""
        popped: list[TinkerSampleRecord] = []
        with self._lock:
            ids = self._episodes.pop(episode_id, None)
            if not ids:
                return []
            while ids:
                completion_id = ids.popleft()
                self._completion_episode.pop(completion_id, None)
                record = self._records.pop(completion_id, None)
                if record:
                    try:
                        self._order.remove(completion_id)
                    except ValueError:
                        pass
                    popped.append(record)
        return popped

    def pop_next(self) -> TinkerSampleRecord | None:
        with self._lock:
            while self._order:
                completion_id = self._order.popleft()
                record = self._records.pop(completion_id, None)
                if record:
                    episode_id = self._completion_episode.pop(completion_id, None)
                    if episode_id:
                        try:
                            self._episodes.get(episode_id, deque()).remove(completion_id)
                        except ValueError:
                            pass
                        if episode_id in self._episodes and not self._episodes[episode_id]:
                            self._episodes.pop(episode_id, None)
                    return record
            return None

    def size(self) -> int:
        with self._lock:
            return len(self._records)


def _data_url_to_pil_image(url: str) -> Image.Image | None:
    """Convert a data URL to a PIL Image."""
    if not url.startswith("data:"):
        return None
    if "," not in url:
        return None
    header, b64_data = url.split(",", 1)
    if ";base64" not in header:
        return None
    try:
        img_bytes = base64.b64decode(b64_data)
    except (ValueError, Exception):
        return None
    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


class TinkerConfig(BaseAgentConfig):
    """Configuration for TinkerAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Tinker"
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str = "qwen3_instruct"
    base_url: str | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.7
    max_context_tokens: int | None = None

    # Tinker-specific options
    sampling_client: Any | None = None  # tinker.SamplingClient
    renderer: Any | None = None  # tinker_cookbook.renderers.Renderer
    tokenizer: Any | None = None  # tinker_cookbook.tokenizer_utils.Tokenizer
    completion_store: TinkerCompletionStore | None = None

    # Computer use settings
    computer_tool_name: str = "computer"


class TinkerCreateParams(BaseCreateParams, TinkerConfig):
    pass


class TinkerAgent(MCPAgent):
    """
    Tinker agent that uses Tinker's sampling API for model inference.

    This agent is designed for reinforcement learning integration where:
    - Prompts, tokens, and logprobs are captured for training
    - Computer use is supported with proper coordinate normalization
    - Compatible with inference-service for rollout generation

    The agent stores completions in a TinkerCompletionStore which can be
    accessed by the RL training loop to construct trajectories.
    """

    metadata: ClassVar[dict[str, Any] | None] = {
        "display_width": computer_settings.TINKER_COMPUTER_WIDTH,
        "display_height": computer_settings.TINKER_COMPUTER_HEIGHT,
    }
    config_cls: ClassVar[type[BaseAgentConfig]] = TinkerConfig

    @with_signature(TinkerCreateParams)
    @classmethod
    def create(cls, **kwargs: Any) -> TinkerAgent:  # pyright: ignore[reportIncompatibleMethodOverride]
        return MCPAgent.create.__func__(cls, **kwargs)  # type: ignore[return-value]

    def __init__(self, params: TinkerCreateParams | None = None, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.config: TinkerConfig

        # Store Tinker components
        self._sampling_client = self.config.sampling_client
        self._renderer = self.config.renderer
        self._tokenizer = self.config.tokenizer
        self._store = self.config.completion_store or TinkerCompletionStore()
        self._computer_tool_name = self.config.computer_tool_name

        # Message history for context
        self._messages: list[dict[str, Any]] = []

    @property
    def completion_store(self) -> TinkerCompletionStore:
        """Access the completion store for RL training."""
        return self._store

    def update_sampling_client(self, client: Any) -> None:
        """Update the sampling client (e.g., after saving new weights)."""
        self._sampling_client = client

    async def get_system_messages(self) -> list[dict[str, Any]]:
        """Get system messages for Tinker."""
        if self.system_prompt:
            return [{"role": "system", "content": self.system_prompt}]
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[dict[str, Any]]:
        """Format MCP content blocks into Tinker-compatible messages."""
        content: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                # Convert base64 image to PIL Image for Tinker
                mime_type = getattr(block, "mimeType", "image/png")
                data_url = f"data:{mime_type};base64,{block.data}"
                pil_image = _data_url_to_pil_image(data_url)
                if pil_image:
                    content.append({"type": "image", "image": pil_image})
                else:
                    content.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )

        if not content:
            content = [{"type": "text", "text": ""}]

        return [{"role": "user", "content": content}]

    def _convert_messages_for_renderer(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert messages to Tinker renderer format."""
        rendered: list[dict[str, Any]] = []
        tool_message = self._build_tool_prompt(tools)

        for msg in messages:
            role = msg.get("role", "user")
            content = self._normalize_content(msg.get("content"))
            rendered_msg: dict[str, Any] = {"role": role, "content": content}

            # Handle tool calls
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                converted_calls = self._convert_tool_calls(tool_calls)
                if converted_calls:
                    rendered_msg["tool_calls"] = converted_calls

            rendered.append(rendered_msg)

        # Add tool prompt to system message
        if tool_message:
            if rendered and rendered[0]["role"] == "system":
                first_content = rendered[0]["content"]
                if isinstance(first_content, str):
                    rendered[0]["content"] = f"{first_content.rstrip()}\n\n{tool_message}"
                elif isinstance(first_content, list):
                    rendered[0]["content"].insert(
                        0, {"type": "text", "text": tool_message + "\n\n"}
                    )
            else:
                rendered.insert(0, {"role": "system", "content": tool_message})

        return rendered

    def _normalize_content(self, content: Any) -> str | list[dict[str, Any]]:
        """Normalize content to string or list of parts."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append({"type": "text", "text": str(block)})
                continue

            block_type = block.get("type")
            if block_type == "text":
                parts.append({"type": "text", "text": str(block.get("text", ""))})
            elif block_type in {"image_url", "image"}:
                url = None
                if block_type == "image_url":
                    image_url = block.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                    elif isinstance(image_url, str):
                        url = image_url
                else:
                    url = block.get("image")

                if isinstance(url, str):
                    pil_image = _data_url_to_pil_image(url)
                    parts.append({"type": "image", "image": pil_image or url})
                elif hasattr(url, "size"):  # PIL Image
                    parts.append({"type": "image", "image": url})
                else:
                    parts.append({"type": "text", "text": "[image]"})
            else:
                parts.append({"type": "text", "text": str(block)})

        return parts if parts else ""

    def _convert_tool_calls(self, tool_calls: Any) -> list[ToolCall] | None:
        """Convert tool calls to Tinker ToolCall format."""
        if not isinstance(tool_calls, list):
            return None

        converted: list[ToolCall] = []
        for call in tool_calls:
            call_id: str | None = None
            name: str | None = None
            arguments: Any = None

            if isinstance(call, ToolCall):
                # Already a ToolCall object, use as-is
                converted.append(call)
                continue

            if isinstance(call, dict):
                call_id = call.get("id")
                if "function" in call and isinstance(call.get("function"), dict):
                    fn = call["function"]
                    name = fn.get("name")
                    arguments = fn.get("arguments")
                else:
                    name = call.get("name")
                    arguments = call.get("args") or call.get("arguments")
            else:
                fn = getattr(call, "function", None)
                if fn is not None:
                    call_id = getattr(call, "id", None)
                    name = getattr(fn, "name", None)
                    arguments = getattr(fn, "arguments", None)

            if not isinstance(name, str) or not name:
                continue

            # Ensure arguments is a JSON string
            if isinstance(arguments, dict):
                arguments_str = json.dumps(arguments)
            elif isinstance(arguments, str):
                arguments_str = arguments
            else:
                arguments_str = "{}"

            converted.append(
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name=name,
                        arguments=arguments_str,
                    ),
                    id=call_id,
                )
            )

        return converted or None

    def _build_tool_prompt(self, tools: list[dict[str, Any]] | None) -> str | None:
        """Build the tool prompt for the model."""
        if not tools:
            return None

        has_computer_tool = any(
            tool.get("function", {}).get("name") == self._computer_tool_name
            for tool in tools
        )

        lines = ["Available tools:", ""]
        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name", "unknown")
            description = fn.get("description", "")
            params = json.dumps(fn.get("parameters", {}), ensure_ascii=False)
            lines.append(f"- {name}: {description}".strip())
            lines.append(f"  args schema: {params}")
        lines.append("")

        # Add grounding instructions for computer tool
        if has_computer_tool:
            lines.extend(self._get_grounding_instructions())
            lines.append("")

        # Tool call format
        call_template = "\n".join(
            [
                "<tool_call>",
                '{"name": "tool_name", "args": {...}}',
                "</tool_call>",
            ]
        )
        lines.append("When you decide to invoke a tool, call it like this:")
        lines.append(call_template)

        return "\n".join(lines)

    def _get_grounding_instructions(self) -> list[str]:
        """Return grounding instructions for the computer tool."""
        return [
            "## Computer Tool Grounding Instructions",
            "",
            "You are a vision-capable agent that can see the screen. When using the computer tool:",
            "",
            "1. LOOK at the screenshot image carefully to identify UI elements.",
            "2. For click, double_click, move, or scroll actions, you MUST provide SEPARATE",
            '   "x" and "y" integer values (NOT an array) for the target pixel coordinates.',
            "3. Estimate coordinates by examining the element's position in the image.",
            "4. The coordinate system starts at (0, 0) in the top-left corner.",
            "",
            "CORRECT format for coordinates (use SEPARATE x and y fields):",
            '  {"type": "click", "x": 475, "y": 325}',
            "",
            "INCORRECT formats (DO NOT use these):",
            '  {"type": "click", "x": [475, 325]}  <- WRONG: x should not be an array',
            '  {"type": "click", "coordinate": [475, 325]}  <- WRONG: use x and y separately',
            "",
            "Examples of CORRECT computer tool calls:",
            "",
            '- Click a button: {"name": "computer", "args": {"type": "click", "x": 475, "y": 325}}',
            '- Double-click: {"name": "computer", "args": {"type": "double_click", "x": 300, "y": 150}}',
            '- Scroll at position: {"name": "computer", "args": {"type": "scroll", "x": 400, "y": 300, "scroll_y": -3}}',
            '- Type text: {"name": "computer", "args": {"type": "type", "text": "hello world"}}',
            '- Press keys: {"name": "computer", "args": {"type": "keypress", "keys": ["ctrl", "c"]}}',
            '- Take screenshot: {"name": "computer", "args": {"type": "screenshot"}}',
            "",
            "IMPORTANT: x and y MUST be separate integer fields, not arrays or combined values.",
        ]

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas in OpenAI format."""
        tool_schemas = super().get_tool_schemas()
        openai_tools = []
        for schema in tool_schemas:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools

    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        """Get response from Tinker sampling API."""
        if self._sampling_client is None:
            raise ValueError(
                "TinkerAgent requires a sampling_client. "
                "Provide it via TinkerConfig or use update_sampling_client()."
            )
        if self._renderer is None:
            raise ValueError(
                "TinkerAgent requires a renderer. "
                "Provide it via TinkerConfig."
            )
        if self._tokenizer is None:
            raise ValueError(
                "TinkerAgent requires a tokenizer. "
                "Provide it via TinkerConfig."
            )

        import tinker

        # Get tool schemas for the prompt
        tools = self.get_tool_schemas()

        # Convert messages to renderer format
        rendered_messages = self._convert_messages_for_renderer(messages, tools)

        # Build prompt with optional context truncation
        max_tokens_value = self.config.max_new_tokens
        prompt = self._build_prompt_with_truncation(
            rendered_messages, max_tokens=max_tokens_value
        )

        # Get stop sequences from renderer
        base_stop = self._renderer.get_stop_sequences()

        # Add </tool_call> to stop sequences to prevent multiple tool calls
        tool_call_end_token = self._tokenizer.encode("</tool_call>")
        if tool_call_end_token and tool_call_end_token[0] not in base_stop:
            base_stop = list(base_stop) + [tool_call_end_token[0]]

        sampling_params = tinker.SamplingParams(
            temperature=self.config.temperature,
            max_tokens=max_tokens_value,
            stop=base_stop,
        )

        result = await self._sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        )

        sequence = result.sequences[0]
        tokens = sequence.tokens
        logprobs = sequence.logprobs
        assistant_message, _ = self._renderer.parse_response(tokens)
        assistant_message = self._normalize_tool_call_text(assistant_message)

        completion_id = f"tinker-{uuid.uuid4().hex}"
        finish_reason = getattr(result, "stop_reason", None) or "stop"
        created_at = time.time()

        # Store completion for RL training
        episode_id = get_current_episode_id()
        self._store.add(
            TinkerSampleRecord(
                completion_id=completion_id,
                prompt=prompt,
                tokens=tokens,
                logprobs=logprobs,
                finish_reason=finish_reason,
                created_at=created_at,
            ),
            episode_id=episode_id,
        )

        # Build response
        content = assistant_message.get("content", "")
        tool_calls: list[MCPToolCall] = []

        if "tool_calls" in assistant_message:
            for idx, call in enumerate(assistant_message["tool_calls"]):
                if isinstance(call, dict):
                    name = call.get("name")
                    args = call.get("args", {})
                    call_id = call.get("id") or f"tool_call_{idx}"
                else:
                    # Handle ToolCall dataclass from renderer
                    name = call.function.name if hasattr(call, "function") else None
                    args_str = call.function.arguments if hasattr(call, "function") else "{}"
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}
                    call_id = getattr(call, "id", None) or f"tool_call_{idx}"

                if name:
                    # Normalize computer tool args
                    if name == self._computer_tool_name:
                        args = self._normalize_computer_tool_args(args)

                    tool_calls.append(
                        MCPToolCall(name=name, arguments=args, id=call_id)
                    )

        # Add assistant message to message history
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments or {}),
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            done=not bool(tool_calls),
            info={"completion_id": completion_id},
        )

    def _build_prompt_with_truncation(
        self, rendered_messages: list[dict[str, Any]], *, max_tokens: int
    ) -> Any:
        """Build prompt with optional context truncation."""
        messages_for_prompt = list(rendered_messages)
        prompt = self._renderer.build_generation_prompt(messages_for_prompt)

        if self.config.max_context_tokens is None:
            return prompt

        # Keep system messages, drop oldest non-system messages as needed
        system_prefix_len = 0
        for msg in messages_for_prompt:
            if msg.get("role") == "system":
                system_prefix_len += 1
            else:
                break

        while (
            prompt.length + max_tokens > self.config.max_context_tokens
            and len(messages_for_prompt) > system_prefix_len + 1
        ):
            messages_for_prompt.pop(system_prefix_len)
            prompt = self._renderer.build_generation_prompt(messages_for_prompt)

        return prompt

    def _normalize_tool_call_text(self, assistant_message: dict[str, Any]) -> dict[str, Any]:
        """Normalize tool call text and parse structured tool_calls."""
        content = assistant_message.get("content")
        if not isinstance(content, str):
            return assistant_message

        # Fix missing opening tag
        if "</tool_call>" in content and "<tool_call>" not in content:
            fixed = "<tool_call>\n" + content
            assistant_message["content"] = fixed
            content = fixed

        # Parse tool call from text if not already parsed
        if "tool_calls" not in assistant_message:
            match = re.search(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return assistant_message
                if isinstance(payload, dict):
                    name = payload.get("name")
                    args = payload.get("args") or payload.get("arguments")
                    if isinstance(name, str) and isinstance(args, dict):
                        args = self._normalize_computer_tool_args(args)
                        assistant_message["tool_calls"] = [
                            {"name": name, "args": args, "id": None}
                        ]

        return assistant_message

    def _normalize_computer_tool_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Normalize coordinate formats for computer tool."""
        args = dict(args)  # Don't mutate the original

        # Handle x being an array of [x, y]
        x_val = args.get("x")
        if isinstance(x_val, (list, tuple)) and len(x_val) >= 2:
            try:
                args["x"] = int(x_val[0])
                args["y"] = int(x_val[1])
            except (TypeError, ValueError, IndexError):
                pass

        # Handle "coordinate" or "coordinates" array
        for coord_key in ("coordinate", "coordinates", "coord", "point"):
            coord = args.get(coord_key)
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                try:
                    args["x"] = int(coord[0])
                    args["y"] = int(coord[1])
                    del args[coord_key]
                except (TypeError, ValueError, IndexError):
                    pass
                break

        # Handle destination coordinates for drag
        for dest_key in ("destination", "destination_coordinate", "end_coordinate"):
            dest = args.get(dest_key)
            if isinstance(dest, (list, tuple)) and len(dest) >= 2:
                try:
                    args["destination_x"] = int(dest[0])
                    args["destination_y"] = int(dest[1])
                    del args[dest_key]
                except (TypeError, ValueError, IndexError):
                    pass
                break

        return args

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        """Format tool results for Tinker messages."""
        rendered: list[dict[str, Any]] = []

        # Separate text and image content
        image_parts: list[dict[str, Any]] = []
        for call, res in zip(tool_calls, tool_results, strict=False):
            text_parts: list[str] = []
            items = res.content
            if not res.content and res.structuredContent:
                items = [res.structuredContent.get("result", res.content)]

            for item in items:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        mime_type = item.get("mimeType", "image/png")
                        data = item.get("data", "")
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{data}"},
                            }
                        )
                elif isinstance(item, types.TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, types.ImageContent):
                    image_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"},
                        }
                    )

            text_content = "".join(text_parts) if text_parts else "Tool executed successfully"
            rendered.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": text_content,
                }
            )

        # Add images as a separate user message
        if image_parts:
            content_with_images = [
                {"type": "text", "text": "Tool returned the following:"},
                image_parts[-1],
            ]
            rendered.append({"role": "user", "content": content_with_images})

        return rendered


# Type exports for external use
__all__ = [
    "TinkerAgent",
    "TinkerConfig",
    "TinkerCreateParams",
    "TinkerCompletionStore",
    "TinkerSampleRecord",
    "get_current_episode_id",
    "set_current_episode_id",
    "reset_current_episode_id",
]
