"""OpenAI-compatible Chat Completions agent — ``ToolAgent`` over chat.completions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from hud.agents.tool_agent import RunState, ToolAgent
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
class OpenAIChatRunState(RunState[ChatCompletionMessageParam]):
    continuation_token_ids: list[int] | None = None
    continuation_message_count: int | None = None


class OpenAIChatAgent(ToolAgent[ChatCompletionMessageParam]):
    """OpenAI-compatible agent using the chat.completions protocol."""

    tool_catalog = (
        ReadTool,
        GrepTool,
        GlobTool,
        ListTool,
        OpenAICompatibleMCPProxyTool,
    )

    def __init__(self, config: OpenAIChatConfig | None = None) -> None:
        config = config or OpenAIChatConfig()
        self.config = config
        self.model = config.model
        self.auto_respond = config.auto_respond
        self.hosted_tools = list(config.hosted_tools)

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

    async def _initialize_state(self, *, prompt: str) -> OpenAIChatRunState:
        return OpenAIChatRunState(
            messages=[
                cast(
                    "ChatCompletionMessageParam",
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ),
            ]
        )

    def _format_user_text(self, text: str) -> ChatCompletionMessageParam:
        return cast(
            "ChatCompletionMessageParam",
            {"role": "user", "content": [{"type": "text", "text": text}]},
        )

    def _format_result(
        self,
        call: MCPToolCall,
        result: MCPToolResult,
        state: RunState[ChatCompletionMessageParam],
    ) -> ChatCompletionMessageParam | list[ChatCompletionMessageParam] | None:
        return format_chat_result(call, result)

    async def get_response(
        self,
        state: RunState[ChatCompletionMessageParam],
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

        try:
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
        except Exception as e:
            error_content = f"Error getting response {e}"
            if "Invalid JSON" in str(e):
                error_content = "Invalid JSON, response was truncated"
            logger.warning(error_content)
            return AgentResponse(
                content=error_content,
                tool_calls=[],
                done=True,
                isError=True,
                raw=None,
            )

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
            if prompt_token_ids is not None and token_ids is not None:
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
            raw_args = json.loads(tc.function.arguments or "{}")
            arguments = cast("dict[str, Any]", raw_args) if isinstance(raw_args, dict) else {}
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
