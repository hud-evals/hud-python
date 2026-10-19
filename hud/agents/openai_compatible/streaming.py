"""Accumulates a streamed chat completion into the ``ChatCompletion`` a plain call returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from openai.types.chat import ChatCompletion

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

_TEXT_MESSAGE_FIELDS = frozenset({"content", "refusal", "reasoning", "reasoning_content"})
# OpenRouter streams reasoning text in pieces but repeats ``type``/``format`` on every
# chunk and sends ``id``/``data``/``signature`` whole.
_TEXT_REASONING_DETAIL_FIELDS = frozenset({"text", "summary"})


@dataclass
class _ChoiceAccumulator:
    message: dict[str, Any] = field(default_factory=lambda: {"role": "assistant"})
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    reasoning_details: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None

    def add(self, choice: ChunkChoice) -> None:
        if choice.finish_reason is not None:
            self.finish_reason = choice.finish_reason
        for key, value in choice.delta.to_dict().items():
            if value is None:
                continue
            if key == "tool_calls":
                for delta in cast("list[dict[str, Any]]", value):
                    self._add_tool_call(delta)
            elif key == "reasoning_details" and isinstance(value, list):
                for delta in cast("list[Any]", value):
                    if isinstance(delta, dict):
                        self._add_reasoning_detail(cast("dict[str, Any]", delta))
            elif key in _TEXT_MESSAGE_FIELDS and isinstance(value, str):
                self.message[key] = self.message.get(key, "") + value
            else:
                self.message[key] = value

    def _add_tool_call(self, delta: dict[str, Any]) -> None:
        call = self.tool_calls.setdefault(delta["index"], {"function": {}})
        for key, value in delta.items():
            if key == "index" or value is None:
                continue
            if key == "function":
                function = call["function"]
                for fn_key, fn_value in cast("dict[str, Any]", value).items():
                    if fn_value is None:
                        continue
                    if fn_key == "arguments":
                        function[fn_key] = function.get(fn_key, "") + fn_value
                    else:
                        function[fn_key] = fn_value
            else:
                call[key] = value

    def _add_reasoning_detail(self, delta: dict[str, Any]) -> None:
        index = delta.get("index")
        if not isinstance(index, int):
            index = len(self.reasoning_details)
        detail = self.reasoning_details.setdefault(index, {})
        for key, value in delta.items():
            if value is None:
                continue
            if key in _TEXT_REASONING_DETAIL_FIELDS and isinstance(value, str):
                detail[key] = detail.get(key, "") + value
            else:
                detail[key] = value

    def to_dict(self, index: int) -> dict[str, Any]:
        message = dict(self.message)
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        if self.reasoning_details:
            message["reasoning_details"] = [
                self.reasoning_details[i] for i in sorted(self.reasoning_details)
            ]
        return {"index": index, "message": message, "finish_reason": self.finish_reason}


class ChatCompletionAccumulator:
    """Folds ``ChatCompletionChunk``s into a ``ChatCompletion``.

    Text fields (message content, refusal, reasoning, reasoning_content, tool-call
    arguments, and reasoning_details text/summary) are concatenated across chunks;
    every other field takes its latest non-null value.
    """

    def __init__(self) -> None:
        self._completion: dict[str, Any] = {}
        self._choices: dict[int, _ChoiceAccumulator] = {}

    def add(self, chunk: ChatCompletionChunk) -> None:
        for key in ("id", "created", "model", "system_fingerprint", "service_tier"):
            if (value := getattr(chunk, key)) is not None:
                self._completion[key] = value
        if chunk.usage is not None:
            self._completion["usage"] = chunk.usage.to_dict()
        for choice in chunk.choices:
            self._choices.setdefault(choice.index, _ChoiceAccumulator()).add(choice)

    def completion(self) -> ChatCompletion:
        if not self._completion:
            raise RuntimeError("Chat completion stream ended without any chunks")
        choices = [self._choices[i].to_dict(i) for i in sorted(self._choices)]
        return ChatCompletion.construct(
            **self._completion, object="chat.completion", choices=choices
        )
