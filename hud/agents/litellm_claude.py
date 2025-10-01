"""LiteLLM-powered Claude MCP Agent.

This mirrors the MCP behavior of hud/agents/claude.py (Anthropic SDK),
but routes requests through LiteLLM (e.g., model="openrouter/anthropic/claude-sonnet-4")
and preserves:
- Anthropic Computer Use tool mapping ("computer_2025-01-24" / "computer_2024-10-22")
- Ephemeral prompt caching blocks ("cache_control": {"type": "ephemeral"})
- The same MCP tool plumbing and result formatting (tool_use <-> tool_result)
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

import litellm
import mcp.types as types

import hud
from hud.settings import settings  # not used here, but kept for parity
from hud.tools.computer.settings import computer_settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult
from hud.utils.hud_console import HUDConsole

from .base import MCPAgent

logger = logging.getLogger(__name__)


class LiteLLMClaudeAgent(MCPAgent):
    """
    Claude-style MCP agent (like hud/agents/claude.py) implemented on LiteLLM.

    Key points:
    - Exposes Anthropic tool specs (computer + function tools) to the model
    - Converts Anthropic tool_use blocks into MCP tool calls
    - Sends MCP tool results back as Anthropic tool_result blocks
    - Uses LiteLLM acompletion() for transport
    """

    metadata: ClassVar[dict[str, Any]] = {
        "display_width": computer_settings.ANTHROPIC_COMPUTER_WIDTH,
        "display_height": computer_settings.ANTHROPIC_COMPUTER_HEIGHT,
    }

    def __init__(
        self,
        *,
        model: str = "openrouter/anthropic/claude-sonnet-4",
        max_tokens: int = 4096,
        use_computer_beta: bool = True,
        completion_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model = model
        self.model_name = self.model
        self.max_tokens = max_tokens
        self.use_computer_beta = use_computer_beta
        self.completion_kwargs = completion_kwargs or {}
        self.hud_console = HUDConsole(logger=logger)

        # Claude tool mapping (Claude tool name -> MCP tool name)
        self._claude_to_mcp_tool_map: dict[str, str] = {}
        self.claude_tools: list[dict] = []

        # Append Claude-ish operating instructions (same spirit as claude.py)
        claude_instructions = """
        You are Claude (via LiteLLM). Be thorough, accurate, and autonomous.

        When using tools:
        1) Think through observations and next steps
        2) Use tools decisively (no confirmation prompts)
        3) Verify outcomes and continue until the task is complete
        4) Prefer minimal steps and clear explanations

        You have access to Anthropic-style tools (including computer use).
        """.strip()

        if self.system_prompt:
            self.system_prompt = f"{self.system_prompt}\n\n{claude_instructions}"
        else:
            self.system_prompt = claude_instructions

    async def initialize(self, task: Any = None) -> None:
        await super().initialize(task)
        self._convert_tools_for_claude()

    #
    # Message formatting (OpenAI-chat-like content blocks that LiteLLM accepts)
    #

    async def get_system_messages(self) -> list[dict[str, Any]]:
        # Anthropic expects the system prompt as a top-level argument, not a chat message.
        return []

    async def format_blocks(self, blocks: list[types.ContentBlock]) -> list[dict[str, Any]]:
        """Convert MCP content blocks to LiteLLM chat message format."""
        parts: list[dict[str, Any]] = []
        for block in blocks:
            if isinstance(block, types.TextContent):
                parts.append({"type": "text", "text": block.text})
            elif isinstance(block, types.ImageContent):
                mime = getattr(block, "mimeType", "image/png")
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": block.data,
                        },
                    }
                )
        return [{"role": "user", "content": parts}]

    #
    # Main step: call LiteLLM and parse tool_use / text / thinking
    #

    @hud.instrument(span_type="agent", record_args=False, record_result=True)
    async def get_response(self, messages: list[dict[str, Any]]) -> AgentResponse:
        current_messages = messages.copy()

        # Add Anthropic ephemeral cache control to the most recent user message blocks
        messages_cached = self._add_prompt_caching(current_messages)

        # Build request
        request_kwargs = self._build_litellm_request(messages_cached)

        attempt_messages = messages_cached
        while True:
            try:
                request_kwargs["messages"] = attempt_messages
                response = await litellm.acompletion(**request_kwargs)
                break
            except Exception as e:
                msg = str(e).lower()
                too_long = (
                    "prompt is too long" in msg
                    or "request_too_large" in msg
                    or "413" in msg
                )
                if too_long and len(attempt_messages) > 21:
                    attempt_messages = [attempt_messages[0], *attempt_messages[-20:]]
                    self.hud_console.warning(
                        "Prompt too long via LiteLLM; truncating history and retrying"
                    )
                    continue
                err = f"Error from LiteLLM: {e}"
                self.hud_console.warning_log(err)
                return AgentResponse(
                    content=err, tool_calls=[], done=True, isError=True, raw=None
                )

        choice = response.choices[0]
        msg = choice.message  # OpenAI-style ChatCompletionMessage proxy from LiteLLM

        # Make sure this assistant message is appended to history for the loop
        assistant_payload: dict[str, Any] = {"role": "assistant"}
        if getattr(msg, "content", None) is not None:
            assistant_payload["content"] = msg.content
        if getattr(msg, "tool_calls", None):
            # Serialize in OpenAI tool_calls shape if present
            serialized = []
            for tc in msg.tool_calls:
                serialized.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            assistant_payload["tool_calls"] = serialized
        messages.append(assistant_payload)

        # Parse result into MCPAgentResponse
        result = AgentResponse(content="", tool_calls=[], done=True, raw=response)

        # Try both shapes:
        # 1) OpenAI-style function `tool_calls`
        # 2) Anthropic-style 'tool_use' blocks inside `content` (for computer use)
        tool_calls_found = False

        # (1) OpenAI / function tools
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                try:
                    args = _safe_json_loads(tc.function.arguments)
                except Exception:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                name = tc.function.name
                if name in self._claude_to_mcp_tool_map:
                    name = self._claude_to_mcp_tool_map[name]
                result.tool_calls.append(MCPToolCall(id=tc.id, name=name, arguments=args))
            tool_calls_found = len(result.tool_calls) > 0

        # (2) Anthropic / tool_use blocks
        # LiteLLM returns assistant.content either as str or list[dict{type,...}]
        content_blocks = getattr(msg, "content", None)
        if isinstance(content_blocks, list):
            thinking_text = ""
            plain_text = ""
            for block in content_blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    # Convert to MCP tool call
                    tool_name = block.get("name", "")
                    mcp_name = self._claude_to_mcp_tool_map.get(tool_name, tool_name)
                    tool_id = block.get("id")
                    tool_input = block.get("input", {})
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
                    tool_calls_found = True
                elif btype == "text":
                    plain_text += block.get("text", "")
                elif btype == "thinking":
                    # Optional: some Anthropic models expose thinking blocks via LiteLLM
                    thinking_text += f"Thinking: {block.get('thinking', '')}\n"

            # Accumulate text/thinking into result.content (when no immediate tool calls)
            if not tool_calls_found:
                result.content = (thinking_text + plain_text).strip()
        else:
            # Assistant content may just be a string
            if isinstance(content_blocks, str) and not tool_calls_found:
                result.content = content_blocks

        # Mark done only if the model did not request tools or we hit length
        result.done = not tool_calls_found or choice.finish_reason == "length"
        return result

    async def format_tool_results(
        self, tool_calls: list[MCPToolCall], tool_results: list[MCPToolResult]
    ) -> list[dict[str, Any]]:
        """
        Return Anthropic tool_result blocks embedded in a single user message.

        LiteLLM passes this through to Anthropic correctly. We keep base64 image
        blocks for screenshots (computer use) and text blocks for logs.
        """
        user_content: list[dict[str, Any]] = []

        for call, res in zip(tool_calls, tool_results, strict=True):
            # Anthropic tool_result wants: {"type": "tool_result", "tool_use_id": <id>, "content": [ ... ]}
            tool_result_blocks: list[dict[str, Any]] = []

            if res.isError:
                error_msg = "Tool execution failed"
                for c in res.content:
                    if isinstance(c, types.TextContent):
                        error_msg = c.text
                        break
                tool_result_blocks.append({"type": "text", "text": f"Error: {error_msg}"})
            else:
                for c in res.content:
                    if isinstance(c, types.TextContent):
                        tool_result_blocks.append({"type": "text", "text": c.text})
                    elif isinstance(c, types.ImageContent):
                        # Anthropic expects "image" with source base64 (not image_url) inside tool_result
                        tool_result_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": getattr(c, "mimeType", "image/png"),
                                    "data": c.data,
                                },
                            }
                        )

            # Only include tool_result if we have a tool_use_id
            if call.id:
                user_content.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": tool_result_blocks}
                )

        # One "user" message containing all tool_result blocks
        return [{"role": "user", "content": user_content}]

    #
    # Helpers
    #

    def _convert_tools_for_claude(self) -> list[dict]:
        """Build Anthropic tool specs from available MCP tools (same logic as hud/agents/claude.py)."""
        claude_tools: list[dict] = []
        self._claude_to_mcp_tool_map.clear()

        # Prefer Anthropic computer MCP tool, with common suffix fallback
        computer_tool_priority = ["anthropic_computer", "computer_anthropic", "computer"]
        selected_computer_tool = None
        for priority_name in computer_tool_priority:
            for t in self._available_tools:
                if t.name == priority_name or t.name.endswith(f"_{priority_name}"):
                    selected_computer_tool = t
                    break
            if selected_computer_tool:
                break

        if selected_computer_tool:
            ctype = self._choose_computer_tool_type()  # "computer_20250124" or "computer_20241022"
            claude_tools.append(
                {
                    "type": ctype,
                    "function": {
                        "name": "computer",
                        "parameters": {
                            "display_width_px": self.metadata["display_width"],
                            "display_height_px": self.metadata["display_height"],
                        },
                    },
                }
            )
            self._claude_to_mcp_tool_map["computer"] = selected_computer_tool.name
            self.hud_console.debug(f"Using '{selected_computer_tool.name}' as MCP computer tool")

        # Add non-computer tools as Anthropic function-style tools (input_schema)
        for t in self._available_tools:
            is_computer = any(
                t.name == p or t.name.endswith(f"_{p}") for p in computer_tool_priority
            )
            if is_computer or t.name in self.lifecycle_tools:
                continue

            claude_tools.append(
                {
                    "name": t.name,
                    "description": t.description or f"Execute {t.name}",
                    "input_schema": t.inputSchema
                    or {
                        "type": "object",
                        "properties": {},
                    },
                }
            )
            self._claude_to_mcp_tool_map[t.name] = t.name

        self.claude_tools = claude_tools
        return claude_tools

    def _choose_computer_tool_type(self) -> str:
        """Heuristic: pick the correct Anthropic computer tool tag based on model."""
        name = (self.model or "").lower()

        # Latest models (Claude 4 Sonnet, Claude 3.7 Sonnet) use 2025-01-24
        newer_markers = ("sonnet-4", "claude-4", "3-7-sonnet", "202502", "202503")
        if any(m in name for m in newer_markers):
            return "computer_20250124"

        # Claude 3.5 Sonnet era
        return "computer_20241022"

    def _build_litellm_request(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Build the kwargs for litellm.acompletion(...) matching the Anthropic MCP flow.
        """
        protected = {"model", "messages", "tools", "tool_choice", "max_tokens"}
        extra = {k: v for k, v in (self.completion_kwargs or {}).items() if k not in protected}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": self.claude_tools,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "max_tokens": self.max_tokens,
            # Be resilient when routing to non-Anthropic backends (OpenRouter, etc.)
            "drop_params": True,
            **extra,
        }

        kwargs["system"] = self.system_prompt

        return kwargs

    def _add_prompt_caching(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Mark the *last user message's* content blocks as ephemeral cacheable for Anthropic.
        (Other providers ignore this safely; LiteLLM drops unsupported params.)
        """
        msgs = list(messages)
        if not msgs:
            return msgs

        last = msgs[-1]
        if last.get("role") != "user":
            return msgs

        content = last.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {
                    "text",
                    "image",
                    "tool_use",
                    "tool_result",
                }:
                    block["cache_control"] = {"type": "ephemeral"}  # type: ignore[typeddict-item]
        return msgs


def _safe_json_loads(s: str | None) -> Any:
    import json

    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}
