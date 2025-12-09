"""Tool format parsing and conversion for OpenAI, Claude, Gemini, and MCP."""

from __future__ import annotations

import json
from enum import Enum, auto
from typing import Any

from hud.types import MCPToolCall, MCPToolResult

__all__ = [
    "ToolFormat",
    "format_result",
    "parse_tool_call",
    "parse_tool_calls",
    "result_to_string",
]


class ToolFormat(Enum):
    """Detected tool call format."""

    OPENAI = auto()  # function.arguments as JSON string
    CLAUDE = auto()  # type="tool_use", input as dict
    GEMINI = auto()  # functionCall with args
    MCP = auto()  # name + arguments


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert object to dict for uniform processing."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise ValueError(f"Cannot convert {type(obj).__name__} to dict")


def _parse_json_args(args: Any) -> dict[str, Any]:
    """Parse arguments, handling JSON strings."""
    if not args:
        return {}
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args


def parse_tool_call(call: Any, **kwargs: Any) -> tuple[MCPToolCall, ToolFormat]:
    """Parse any tool call format into (MCPToolCall, ToolFormat).

    Supports:
        - String (tool name only, or with kwargs)
        - Tuple: (name,), (name, args), (name, args, id)
        - MCPToolCall
        - OpenAI: {function: {name, arguments}, id}
        - Claude: {type: "tool_use", name, input, id}
        - Gemini: {functionCall: {name, args}} or {name, args}
        - Generic: {name, arguments}

    Args:
        call: Tool call in any supported format.
        **kwargs: Additional arguments (merged when call is a string).

    Returns:
        Tuple of (MCPToolCall, ToolFormat) for the parsed call.

    Raises:
        ValueError: If format is unrecognized.
    """
    # Primitives
    if isinstance(call, str):
        return MCPToolCall(name=call, arguments=kwargs or {}), ToolFormat.MCP

    if isinstance(call, tuple):
        tc = MCPToolCall(name=call[0], arguments=call[1] if len(call) > 1 else {})
        if len(call) > 2:
            tc.id = call[2]
        return tc, ToolFormat.MCP

    if isinstance(call, MCPToolCall):
        return call, ToolFormat.MCP

    # Convert to dict
    d = _to_dict(call)

    # OpenAI: {function: {name, arguments}, id}
    if "function" in d:
        f = _to_dict(d["function"]) if not isinstance(d["function"], dict) else d["function"]
        tc = MCPToolCall(name=f["name"], arguments=_parse_json_args(f.get("arguments")))
        if d.get("id"):
            tc.id = d["id"]
        return tc, ToolFormat.OPENAI

    # Claude: {type: "tool_use", name, input, id}
    if d.get("type") == "tool_use":
        tc = MCPToolCall(name=d["name"], arguments=d.get("input") or {})
        if d.get("id"):
            tc.id = d["id"]
        return tc, ToolFormat.CLAUDE

    # Gemini: {functionCall: {name, args}} or {name, args}
    if "functionCall" in d:
        fc = d["functionCall"]
        return MCPToolCall(name=fc["name"], arguments=fc.get("args") or {}), ToolFormat.GEMINI

    if "args" in d and "name" in d and "arguments" not in d:
        return MCPToolCall(name=d["name"], arguments=d.get("args") or {}), ToolFormat.GEMINI

    # Generic: {name, arguments/input}
    if "name" in d:
        tc = MCPToolCall(name=d["name"], arguments=d.get("arguments") or d.get("input") or {})
        if d.get("id"):
            tc.id = d["id"]
        return tc, ToolFormat.MCP

    raise ValueError(f"Unrecognized tool call format: {list(d.keys())}")


def _is_tool_block(item: Any) -> bool:
    """Check if item is a tool call (not text/other content)."""
    t = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
    return t is None or t in ("tool_use", "function")


def parse_tool_calls(calls: Any) -> list[tuple[MCPToolCall, ToolFormat]]:
    """Parse multiple tool calls, filtering non-tool content (e.g. Claude TextBlock).

    Args:
        calls: Single call or list of calls in any format.

    Returns:
        List of (MCPToolCall, ToolFormat) tuples.
    """
    if calls is None:
        return []
    if not isinstance(calls, list):
        try:
            return [parse_tool_call(calls)]
        except ValueError:
            return []

    results = []
    for item in calls:
        if not _is_tool_block(item):
            continue
        try:
            results.append(parse_tool_call(item))
        except ValueError:
            continue
    return results


# -----------------------------------------------------------------------------
# Result Formatting
# -----------------------------------------------------------------------------


def result_to_string(result: MCPToolResult) -> str:
    """Convert MCPToolResult content to string.

    Args:
        result: MCP tool result with content blocks.

    Returns:
        String representation of the result content.
    """
    if not result.content:
        return ""
    parts = []
    for block in result.content:
        if (text := getattr(block, "text", None)) is not None:
            parts.append(str(text))
        elif (data := getattr(block, "data", None)) is not None:
            parts.append(f"[binary: {len(data)} bytes]")
    return "\n".join(parts)


def format_result(result: MCPToolResult, tc: MCPToolCall, fmt: ToolFormat) -> Any:
    """Format MCPToolResult based on the input format.

    Args:
        result: MCP tool result.
        tc: Original tool call (for id/name).
        fmt: Target format.

    Returns:
        OpenAI: {"role": "tool", "tool_call_id": ..., "content": ...}
        Claude: {"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error"?: bool}
        Gemini: {"functionResponse": {"name": ..., "response": {"result": ...}}}
        MCP: MCPToolResult unchanged
    """
    content = result_to_string(result)

    if fmt == ToolFormat.OPENAI:
        return {"role": "tool", "tool_call_id": tc.id, "content": content}

    if fmt == ToolFormat.CLAUDE:
        r: dict[str, Any] = {"type": "tool_result", "tool_use_id": tc.id, "content": content}
        if result.isError:
            r["is_error"] = True
        return r

    if fmt == ToolFormat.GEMINI:
        return {"functionResponse": {"name": tc.name, "response": {"result": content}}}

    return result  # MCP format - return as-is
