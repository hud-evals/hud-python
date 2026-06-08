"""OpenAI-compatible Chat Completions agent — ``ToolAgent`` over chat.completions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.tools.base import parse_tool_arguments
from hud.agents.types import OpenAIChatConfig
from hud.settings import settings
from hud.types import AgentResponse, MCPToolCall, MCPToolResult, Sample

from .tools import (
    GlobTool,
    GrepTool,
    ListTool,
    OpenAICompatibleMCPProxyTool,
    ReadTool,
)
from .tools.base import format_chat_result

logger = logging.getLogger(__name__)


@dataclass
class OpenAIChatRunState(RunState[ChatCompletionMessageParam, ChatCompletionToolParam]):
    continuation_token_ids: list[int] | None = None
    continuation_message_count: int | None = None


class OpenAIChatAgent(
    ToolAgent[ChatCompletionMessageParam, ChatCompletionToolParam, OpenAIChatConfig]
):
    """OpenAI-compatible agent using the chat.completions protocol."""

    tools = (ReadTool, GrepTool, GlobTool, ListTool, OpenAICompatibleMCPProxyTool)

    def __init__(self, config: OpenAIChatConfig | None = None) -> None:
        config = config or OpenAIChatConfig()
        self.config = config

        if (
            config.api_key
            and config.base_url
            and settings.hud_gateway_url in config.base_url
            and settings.api_key
            and config.api_key != settings.api_key
        ):
            raise ValueError(
                "OpenAIChatAgent api_key is not allowed with HUD Gateway. "
                "Use HUD_API_KEY for gateway auth and BYOK headers for provider keys."
            )

        self.oai: AsyncOpenAI
        if config.openai_client is not None:
            self.oai = config.openai_client
        elif config.api_key is not None or config.base_url is not None:
            self.oai = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        elif settings.api_key:
            self.oai = AsyncOpenAI(
                api_key=settings.api_key,
                base_url=settings.hud_gateway_url,
            )
        else:
            raise ValueError(
                "No API key found. Set HUD_API_KEY for HUD gateway, "
                "or provide api_key/base_url/openai_client explicitly."
            )

        self.completion_kwargs = dict(config.completion_kwargs)
        if config.checkpoint:
            extra_body: dict[str, Any] = dict(self.completion_kwargs.get("extra_body") or {})
            extra_body["checkpoint"] = config.checkpoint
            self.completion_kwargs["extra_body"] = extra_body

    # ─── ToolAgent hooks ──────────────────────────────────────────────

    async def _initialize_state(self, *, prompt: str | list[Any] | None) -> OpenAIChatRunState:
        return OpenAIChatRunState(messages=self._initial_messages(prompt))

    def _format_message(self, role: str, text: str) -> ChatCompletionMessageParam:
        return cast(
            "ChatCompletionMessageParam",
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": [{"type": "text", "text": text}],
            },
        )

    def _format_result(
        self,
        call: MCPToolCall,
        result: MCPToolResult,
        state: RunState[ChatCompletionMessageParam, ChatCompletionToolParam],
    ) -> ChatCompletionMessageParam | list[ChatCompletionMessageParam] | None:
        return format_chat_result(call, result)

    async def get_response(
        self,
        state: RunState[ChatCompletionMessageParam, ChatCompletionToolParam],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentResponse:
        del citations_enabled
        chat_state = cast("OpenAIChatRunState", state)
        messages = chat_state.messages

        reserved_kwargs = {"model", "messages", "stream", "tools"}
        request_kwargs = {
            key: value
            for key, value in self.completion_kwargs.items()
            if key not in reserved_kwargs
        }
        provider_body: dict[str, Any] = dict(request_kwargs.pop("extra_body", None) or {})
        return_token_ids = bool(provider_body.get("return_token_ids"))

        if state.params:
            provider_body["tools"] = state.params

        if (
            return_token_ids
            and chat_state.continuation_token_ids
            and chat_state.continuation_message_count
        ):
            provider_body["prompt_token_ids"] = chat_state.continuation_token_ids
            provider_body["continuation_from"] = chat_state.continuation_message_count

        if provider_body:
            request_kwargs["extra_body"] = provider_body

        # Token ids imply training intent → also collect per-token sampling logprobs.
        if return_token_ids:
            request_kwargs.setdefault("logprobs", True)

        response: ChatCompletion = await self.oai.chat.completions.create(
            model=self.model,
            messages=(
                [{"role": "system", "content": system_prompt}, *messages]
                if system_prompt is not None
                else messages
            ),
            stream=False,
            **request_kwargs,
        )

        if not response.choices:
            raise ValueError("chat completion returned no choices")
        choice = response.choices[0]
        message = choice.message
        function_calls = [tc for tc in message.tool_calls or [] if tc.type == "function"]

        assistant_message = message.model_dump(exclude_none=True)
        reasoning_content = getattr(message, "reasoning_content", None)
        reasoning = reasoning_content if isinstance(reasoning_content, str) else None
        if not reasoning:
            raw_reasoning = getattr(message, "reasoning", None)
            reasoning = raw_reasoning if isinstance(raw_reasoning, str) else None
        for field_name in ("reasoning_content", "reasoning", "reasoning_details"):
            if value := getattr(message, field_name, None):
                assistant_message[field_name] = value
        if function_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in function_calls
            ]
        messages.append(cast("ChatCompletionMessageParam", assistant_message))

        sample: Sample | None = None
        if return_token_ids:
            prompt_token_ids = getattr(choice, "prompt_token_ids", None)
            token_ids = getattr(choice, "token_ids", None)
            if prompt_token_ids is None or token_ids is None:
                raise ValueError(
                    "return_token_ids was requested but the response is missing "
                    "prompt_token_ids/token_ids; the endpoint does not support token-id "
                    "continuation required for training.",
                )
            chat_state.continuation_token_ids = list(prompt_token_ids) + list(token_ids)
            chat_state.continuation_message_count = len(messages)
            content_lp = choice.logprobs.content if choice.logprobs else None
            sample = Sample(
                prompt_token_ids=list(prompt_token_ids),
                output_token_ids=list(token_ids),
                output_logprobs=[tok.logprob for tok in content_lp] if content_lp else [],
            )

        tool_calls: list[MCPToolCall] = []
        for tc in function_calls:
            provider_name = tc.function.name
            arguments = parse_tool_arguments(tc.function.arguments, provider_name)
            tool_calls.append(
                MCPToolCall(id=tc.id, name=provider_name, arguments=arguments),
            )

        return AgentResponse(
            content=message.content or "",
            reasoning=reasoning,
            finish_reason=choice.finish_reason,
            refusal=message.refusal,
            tool_calls=tool_calls,
            done=not tool_calls,
            raw=response,
            sample=sample,
        )


__all__ = ["OpenAIChatAgent"]
