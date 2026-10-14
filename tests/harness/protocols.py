"""What each provider rejects about a request's conversation, so the fake rejects it too.

Each check returns the reasons the real API would answer 400. They cover the
invariants an agent can break while assembling history: calls answered by the
results right after them, results that answer a real call, thinking that keeps
its signature, and images that decode as the type they claim.
"""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

from PIL import Image, UnidentifiedImageError

IMAGE_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG", "image/gif": "GIF", "image/webp": "WEBP"}
MAX_SIDE = 8000


def image_problem(media_type: str, data: str) -> str | None:
    """Why base64 ``data`` is not an image of ``media_type`` a provider would accept."""
    if media_type not in IMAGE_TYPES:
        return f"unsupported image media type {media_type!r}"
    try:
        raw = base64.b64decode(data, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            kind, size = image.format, image.size
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError):
        return f"image data is not a valid {media_type}"
    if kind != IMAGE_TYPES[media_type]:
        return f"image is {kind} but its media type says {media_type}"
    if max(size) > MAX_SIDE:
        return f"image dimensions exceed {MAX_SIDE} pixels"
    return None


def anthropic(body: dict[str, Any]) -> list[str]:
    messages = body.get("messages") or []
    problems: list[str] = []
    if not messages or messages[0].get("role") != "user":
        problems.append("the first message must be a user message")
    for index, message in enumerate(messages):
        blocks = [block for block in message.get("content") or [] if isinstance(block, dict)]
        if message.get("role") == "assistant":
            calls = {block["id"] for block in blocks if block.get("type") == "tool_use"}
            following = messages[index + 1] if index + 1 < len(messages) else {}
            answered = {
                block.get("tool_use_id")
                for block in following.get("content") or []
                if isinstance(block, dict) and block.get("type") == "tool_result"
            }
            if calls - answered:
                problems.append(
                    f"tool_use ids {sorted(calls - answered)} have no tool_result right after them"
                )
            problems += [
                "a thinking block is missing its signature"
                for block in blocks
                if block.get("type") == "thinking" and not block.get("signature")
            ]
        else:
            previous = messages[index - 1] if index else {}
            asked = {
                block.get("id")
                for block in previous.get("content") or []
                if isinstance(block, dict) and block.get("type") == "tool_use"
            }
            problems += [
                f"tool_result {block.get('tool_use_id')!r} answers no tool_use before it"
                for block in blocks
                if block.get("type") == "tool_result" and block.get("tool_use_id") not in asked
            ]
        for block in _nested(blocks):
            source = block.get("source") or {}
            if block.get("type") == "image" and source.get("type") == "base64":
                problem = image_problem(source.get("media_type", ""), source.get("data", ""))
                problems += [problem] if problem else []
    return problems


def chat(body: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    pending: set[str] = set()
    for message in body.get("messages") or []:
        role = message.get("role")
        if role == "tool":
            if message.get("tool_call_id") not in pending:
                problems.append(
                    f"tool message {message.get('tool_call_id')!r} answers no pending tool call"
                )
            pending.discard(message.get("tool_call_id"))
            continue
        if pending:
            problems.append(
                f"tool calls {sorted(pending)} are not answered before the next message"
            )
            pending = set()
        if role == "assistant":
            pending = {call["id"] for call in message.get("tool_calls") or []}
    if pending:
        problems.append(f"tool calls {sorted(pending)} are not answered")
    return problems


def responses(body: dict[str, Any], issued: dict[str, set[str]]) -> list[str]:
    """``issued`` maps each response id the fake returned to the call ids it carried."""
    previous = body.get("previous_response_id")
    if previous is not None and previous not in issued:
        return [f"previous_response_id {previous!r} was never returned"]
    items = body.get("input")
    items = items if isinstance(items, list) else []
    calls = set(issued.get(previous, set())) | {
        item["call_id"] for item in items if item.get("type", "").endswith("_call")
    }
    answered = [item.get("call_id") for item in items if item.get("type", "").endswith("_output")]
    problems = [f"no tool call found for output {call!r}" for call in answered if call not in calls]
    if previous is not None:
        problems += [
            f"no tool output found for call {call!r}"
            for call in sorted(issued[previous] - set(answered))
        ]
    return problems


def gemini(body: dict[str, Any]) -> list[str]:
    contents = body.get("contents") or []
    problems = [
        f"content role {content.get('role')!r} is neither user nor model"
        for content in contents
        if content.get("role") not in {"user", "model"}
    ]
    for index, content in enumerate(contents):
        called = sum("functionCall" in part for part in content.get("parts") or [])
        if content.get("role") != "model" or not called:
            continue
        following = contents[index + 1] if index + 1 < len(contents) else {}
        answered = sum("functionResponse" in part for part in following.get("parts") or [])
        if following.get("role") != "user" or answered != called:
            problems.append(
                f"{called} function calls are followed by {answered} function responses"
            )
    return problems


def _nested(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``blocks`` and the blocks inside each tool result."""
    found = list(blocks)
    for block in blocks:
        if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
            found += [inner for inner in block["content"] if isinstance(inner, dict)]
    return found
