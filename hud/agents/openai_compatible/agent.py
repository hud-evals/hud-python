"""OpenAI-compatible Chat Completions agent — ``ToolAgent`` over chat.completions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from hud.agents.tool_agent import RunState, ToolAgent
from hud.agents.types import AgentStep, OpenAIChatConfig, Sample, Usage
from hud.settings import settings
from hud.types import MCPToolCall, MCPToolResult
from hud.utils import gateway

from .tools import (
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    OpenAICompatibleMCPProxyTool,
    ReadTool,
    WriteTool,
)
from .tools.base import format_chat_result

if TYPE_CHECKING:
    import mcp.types as mcp_types

logger = logging.getLogger(__name__)


@dataclass
class OpenAIChatRunState(RunState[ChatCompletionMessageParam]):
    continuation_token_ids: list[int] | None = None
    continuation_message_count: int | None = None


class OpenAIChatAgent(ToolAgent[ChatCompletionMessageParam, OpenAIChatConfig]):
    """OpenAI-compatible agent using the chat.completions protocol."""

    tool_catalog = (
        BashTool,
        ReadTool,
        GlobTool,
        GrepTool,
        EditTool,
        WriteTool,
        OpenAICompatibleMCPProxyTool,
    )

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
        if config.model_client is not None:
            self.oai = config.model_client
        elif config.api_key is not None or config.base_url is not None:
            self.oai = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        elif settings.api_key:
            self.oai = cast("AsyncOpenAI", gateway.build_gateway_client("openai"))
        else:
            raise ValueError(
                "No API key found. Set HUD_API_KEY for HUD gateway, "
                "or provide api_key/base_url/model_client explicitly."
            )

        self.completion_kwargs = dict(config.completion_kwargs)
        if config.checkpoint:
            extra_body: dict[str, Any] = dict(self.completion_kwargs.get("extra_body") or {})
            extra_body["checkpoint"] = config.checkpoint
            self.completion_kwargs["extra_body"] = extra_body

    # ─── ToolAgent hooks ──────────────────────────────────────────────

    async def _initialize_state(
        self, *, prompt: list[mcp_types.PromptMessage]
    ) -> OpenAIChatRunState:
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
        state: RunState[ChatCompletionMessageParam],
    ) -> ChatCompletionMessageParam | list[ChatCompletionMessageParam] | None:
        return format_chat_result(call, result)

    async def get_response(
        self,
        state: RunState[ChatCompletionMessageParam],
        *,
        system_prompt: str | None = None,
        citations_enabled: bool = False,
    ) -> AgentStep:
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
                model=self.config.model,
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
            return AgentStep(error=error_content, done=True)

        choice = response.choices[0]
        message = choice.message
        function_calls = [tc for tc in message.tool_calls or [] if tc.type == "function"]
        tool_calls: list[MCPToolCall] = []
        recorded_calls: list[dict[str, Any]] = []
        for tc in function_calls:
            name, raw = tc.function.name, tc.function.arguments
            arguments: dict[str, Any] | str
            try:
                parsed = json.loads(raw or "{}")
                arguments = cast("dict[str, Any]", parsed) if isinstance(parsed, dict) else {}
                recorded = raw
            except json.JSONDecodeError as e:
                logger.warning("Malformed tool-call arguments for %s: %s", name, e)
                arguments = raw
                recorded = "{}"  # replayed history must stay parseable
            tool_calls.append(MCPToolCall(id=tc.id, name=name, arguments=arguments))
            recorded_calls.append(
                {"id": tc.id, "type": "function", "function": {"name": name, "arguments": recorded}}
            )

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
            assistant_message["tool_calls"] = recorded_calls
        messages.append(cast("ChatCompletionMessageParam", assistant_message))

        sample: Sample | None = None
        if return_token_ids:
            prompt_token_ids = getattr(choice, "prompt_token_ids", None)
            # Multimodal prompt (text + image chunks): the only prompt representation
            # that survives image inputs; flat prompt_token_ids is null in that case.
            prompt_chunks = getattr(choice, "prompt_chunks", None)
            token_ids = getattr(choice, "token_ids", None)
            has_prompt = prompt_token_ids is not None or prompt_chunks is not None
            if token_ids is not None and has_prompt:
                content_lp = choice.logprobs.content if choice.logprobs else None
                sample = Sample(
                    prompt_token_ids=list(prompt_token_ids) if prompt_token_ids is not None else [],
                    prompt_chunks=list(prompt_chunks) if prompt_chunks is not None else None,
                    output_token_ids=list(token_ids),
                    output_logprobs=[tok.logprob for tok in content_lp] if content_lp else [],
                )
                # KV-cache continuation only applies to flat text prompts; clear any
                # stale state when the gateway returns chunks-only (multimodal turn).
                if prompt_token_ids is not None:
                    chat_state.continuation_token_ids = list(prompt_token_ids) + list(token_ids)
                    chat_state.continuation_message_count = len(messages)
                else:
                    chat_state.continuation_token_ids = None
                    chat_state.continuation_message_count = None

        usage: Usage | None = None
        if response.usage is not None:
            details = response.usage.prompt_tokens_details
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=details.cached_tokens if details is not None else None,
            )
        return AgentStep(
            content=message.content or "",
            reasoning=reasoning,
            finish_reason=choice.finish_reason,
            refusal=message.refusal,
            tool_calls=tool_calls,
            done=not tool_calls,
            raw=response,
            sample=sample,
            model=response.model,
            usage=usage,
        )


__all__ = ["OpenAIChatAgent"]
