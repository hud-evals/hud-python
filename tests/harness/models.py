"""Scripted model providers behind the fake inference gateway.

One provider-neutral script answers the four wire protocols hud's agents speak:

- OpenAI Chat Completions: ``POST /chat/completions``
- OpenAI Responses: ``POST /responses``
- Anthropic Messages, streamed as server-sent events: ``POST /v1/messages``
- Gemini: ``POST /v1beta/models/{model}:generateContent``

A script is a list of turns, or a function from :class:`ModelRequest` to a turn.
The turn a request gets is read from the conversation the request carries (how
many assistant turns it already holds), so one script serves many concurrent
rollouts. Agents reach the script through ``HUD_GATEWAY_URL`` or a ``base_url``.
"""

from __future__ import annotations

import itertools
import json
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .services import Reply

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from .services import FakeServices, Request


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True)
class Turn:
    """One assistant turn. ``fail`` answers with an HTTP error instead, such as a 529 or 400."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None
    truncated: bool = False
    fail: Reply | None = None


def say(text: str, *, reasoning: str | None = None) -> Turn:
    return Turn(text=text, reasoning=reasoning)


def call(name: str, /, **arguments: Any) -> Turn:
    return Turn(tool_calls=(ToolCall(name, arguments),))


def calls(*tool_calls: ToolCall, text: str | None = None) -> Turn:
    return Turn(text=text, tool_calls=tool_calls)


def fail(status: int, message: str = "scripted provider error") -> Turn:
    return Turn(fail=Reply(status=status, json={"error": {"type": "error", "message": message}}))


@dataclass(frozen=True)
class ModelRequest:
    """A model request as the provider received it, with the turn it asks for."""

    protocol: str
    model: str
    body: dict[str, Any]
    turn: int
    headers: dict[str, str]

    @property
    def tools(self) -> list[str]:
        """Names of the tools the request offers."""
        names: list[str] = []
        for tool in self.body.get("tools") or []:
            if self.protocol == "gemini":
                names.extend(decl["name"] for decl in tool.get("functionDeclarations", []))
            elif self.protocol == "chat":
                names.append(tool["function"]["name"])
            elif "name" in tool:
                names.append(tool["name"])
        return names

    @property
    def tool_results(self) -> list[str]:
        """Texts of the tool results this request returns to the model, oldest first."""
        if self.protocol == "chat":
            return [
                _text(message.get("content"))
                for message in self.body.get("messages", [])
                if message.get("role") == "tool"
            ]
        if self.protocol == "anthropic":
            return [
                _text(block.get("content"))
                for message in self.body.get("messages", [])
                for block in message.get("content") or []
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
        if self.protocol == "responses":
            return [
                _text(item.get("output"))
                for item in _input_items(self.body)
                if item.get("type", "").endswith("_call_output")
            ]
        return [
            json.dumps(part["functionResponse"].get("response"))
            for content in self.body.get("contents", [])
            for part in content.get("parts", [])
            if "functionResponse" in part
        ]

    @property
    def prompt(self) -> str:
        """Text of the first user turn."""
        if self.protocol in {"chat", "anthropic"}:
            for message in self.body.get("messages", []):
                if message.get("role") == "user":
                    return _text(message.get("content"))
        if self.protocol == "responses":
            for item in _input_items(self.body):
                if item.get("role") == "user":
                    return _text(item.get("content"))
        if self.protocol == "gemini":
            for content in self.body.get("contents", []):
                if content.get("role") == "user":
                    return "".join(part.get("text", "") for part in content.get("parts", []))
        return ""


class Models:
    """The scripted providers on a :class:`FakeServices` gateway. Use the ``models`` fixture."""

    def __init__(self, services: FakeServices) -> None:
        self._services = services
        self._turns: list[Turn] = []
        self._policy: Callable[[ModelRequest], Turn] | None = None
        self._responses: dict[str, int] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        services.route("gateway", "POST", "/chat/completions", handler=self._chat)
        services.route("gateway", "POST", "/responses", handler=self._responses_api)
        services.route("gateway", "POST", "/v1/messages", handler=self._anthropic)
        services.route(
            "gateway", "POST", "/v1beta/models/{model}:generateContent", handler=self._gemini
        )

    @property
    def url(self) -> str:
        return self._services.url("gateway")

    def script(self, turns: Sequence[Turn]) -> None:
        """Answer the n-th assistant turn of every conversation with ``turns[n]``."""
        self._turns, self._policy = list(turns), None

    def respond(self, policy: Callable[[ModelRequest], Turn]) -> None:
        """Answer every request with ``policy(request)``."""
        self._turns, self._policy = [], policy

    def requests(self) -> list[ModelRequest]:
        """Every model request so far, decoded, in arrival order."""
        return [self._decode(request) for request in self._services.requests("gateway", "POST")]

    # ─── protocol endpoints ───────────────────────────────────────────

    def _next(self, request: ModelRequest) -> Turn:
        if self._policy is not None:
            return self._policy(request)
        if request.turn >= len(self._turns):
            raise AssertionError(
                f"model script has {len(self._turns)} turns; request asked for turn "
                f"{request.turn + 1} ({request.protocol}, tool results {request.tool_results})"
            )
        return self._turns[request.turn]

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._ids)}"

    def _decode(self, raw: Request) -> ModelRequest:
        body = raw.json or {}
        if raw.path == "/chat/completions":
            protocol, model = "chat", body.get("model", "")
            turn = sum(1 for m in body.get("messages", []) if m.get("role") == "assistant")
        elif raw.path == "/v1/messages":
            protocol, model = "anthropic", body.get("model", "")
            turn = sum(1 for m in body.get("messages", []) if m.get("role") == "assistant")
        elif raw.path == "/responses":
            protocol, model = "responses", body.get("model", "")
            previous = body.get("previous_response_id")
            with self._lock:
                turn = (
                    self._responses[previous] + 1
                    if previous is not None
                    else sum(
                        1
                        for item in _input_items(body)
                        if item.get("role") == "assistant" or item.get("type") == "function_call"
                    )
                )
        else:
            protocol, model = "gemini", raw.params["model"]
            turn = sum(1 for c in body.get("contents", []) if c.get("role") == "model")
        return ModelRequest(protocol, model, body, turn, raw.headers)

    def _chat(self, raw: Request) -> Reply:
        request = self._decode(raw)
        body = request.body
        turn = self._next(request)
        if turn.fail is not None:
            return turn.fail
        tool_calls = [
            {
                "id": tool.id or self._id("call"),
                "type": "function",
                "function": {"name": tool.name, "arguments": json.dumps(tool.arguments)},
            }
            for tool in turn.tool_calls
        ]
        message: dict[str, Any] = {"role": "assistant", "content": turn.text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if turn.reasoning is not None:
            message["reasoning_content"] = turn.reasoning
        finish = "length" if turn.truncated else ("tool_calls" if tool_calls else "stop")
        return Reply(
            json={
                "id": self._id("chatcmpl"),
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", ""),
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish, "logprobs": None}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def _responses_api(self, raw: Request) -> Reply:
        request = self._decode(raw)
        body = request.body
        turn = self._next(request)
        if turn.fail is not None:
            return turn.fail
        response_id = self._id("resp")
        with self._lock:
            self._responses[response_id] = request.turn
        output: list[dict[str, Any]] = []
        if turn.reasoning is not None:
            output.append(
                {
                    "type": "reasoning",
                    "id": self._id("rs"),
                    "summary": [{"type": "summary_text", "text": turn.reasoning}],
                }
            )
        if turn.text is not None:
            output.append(
                {
                    "type": "message",
                    "id": self._id("msg"),
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": turn.text, "annotations": []}],
                }
            )
        output.extend(
            {
                "type": "function_call",
                "id": self._id("fc"),
                "call_id": tool.id or self._id("call"),
                "name": tool.name,
                "arguments": json.dumps(tool.arguments),
                "status": "completed",
            }
            for tool in turn.tool_calls
        )
        return Reply(
            json={
                "id": response_id,
                "object": "response",
                "created_at": 0,
                "status": "incomplete" if turn.truncated else "completed",
                "incomplete_details": {"reason": "max_output_tokens"} if turn.truncated else None,
                "model": body.get("model", ""),
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    def _anthropic(self, raw: Request) -> Reply:
        request = self._decode(raw)
        body = request.body
        turn = self._next(request)
        if turn.fail is not None:
            return turn.fail
        blocks: list[dict[str, Any]] = []
        if turn.reasoning is not None:
            blocks.append({"type": "thinking", "thinking": turn.reasoning, "signature": "sig"})
        if turn.text is not None:
            blocks.append({"type": "text", "text": turn.text})
        blocks.extend(
            {
                "type": "tool_use",
                "id": tool.id or self._id("toolu"),
                "name": tool.name,
                "input": tool.arguments,
            }
            for tool in turn.tool_calls
        )
        stop = "max_tokens" if turn.truncated else ("tool_use" if turn.tool_calls else "end_turn")
        message = {
            "id": self._id("msg"),
            "type": "message",
            "role": "assistant",
            "model": body.get("model", ""),
            "content": blocks,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if not body.get("stream"):
            return Reply(json=message)
        return Reply(stream=_anthropic_events(message), content_type="text/event-stream")

    def _gemini(self, raw: Request) -> Reply:
        turn = self._next(self._decode(raw))
        if turn.fail is not None:
            return turn.fail
        parts: list[dict[str, Any]] = []
        if turn.reasoning is not None:
            parts.append({"text": turn.reasoning, "thought": True})
        if turn.text is not None:
            parts.append({"text": turn.text})
        parts.extend(
            {"functionCall": {"name": tool.name, "args": tool.arguments}}
            for tool in turn.tool_calls
        )
        return Reply(
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": parts},
                        "finishReason": "MAX_TOKENS" if turn.truncated else "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 2,
                },
                "modelVersion": raw.params["model"],
            }
        )


def _anthropic_events(message: dict[str, Any]) -> Iterator[bytes]:
    def event(kind: str, data: dict[str, Any]) -> bytes:
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n".encode()

    start = {**message, "content": [], "stop_reason": None}
    yield event("message_start", {"message": start})
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            yield event(
                "content_block_start",
                {"index": index, "content_block": {"type": "text", "text": ""}},
            )
            yield event(
                "content_block_delta",
                {"index": index, "delta": {"type": "text_delta", "text": block["text"]}},
            )
        elif block["type"] == "thinking":
            yield event(
                "content_block_start",
                {
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            )
            yield event(
                "content_block_delta",
                {
                    "index": index,
                    "delta": {"type": "thinking_delta", "thinking": block["thinking"]},
                },
            )
            yield event(
                "content_block_delta",
                {
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": block["signature"]},
                },
            )
        else:
            yield event(
                "content_block_start",
                {"index": index, "content_block": {**block, "input": {}}},
            )
            yield event(
                "content_block_delta",
                {
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block["input"]),
                    },
                },
            )
        yield event("content_block_stop", {"index": index})
    yield event(
        "message_delta",
        {
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
    )
    yield event("message_stop", {})


def _input_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    items = body.get("input", [])
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    return [item for item in items if isinstance(item, dict)]


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return json.dumps(content)
